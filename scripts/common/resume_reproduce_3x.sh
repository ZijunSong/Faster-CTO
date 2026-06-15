#!/bin/bash
# Resume RSE reproduce pipeline: skip completed runs/steps, continue unfinished work.
# Required env: GPU_ID, MODEL_NAME, DATASET_NAME, QUESTION_FILE, END_INDEX, RUN_TAG
# Optional env: START_RUN (default 0), MAX_MODEL_LEN, MAX_TOKENS, TENSOR_PARALLEL_SIZE
set -e

: "${GPU_ID:?GPU_ID is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${DATASET_NAME:?DATASET_NAME is required}"
: "${QUESTION_FILE:?QUESTION_FILE is required}"
: "${END_INDEX:?END_INDEX is required}"
: "${RUN_TAG:?RUN_TAG is required}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
START_RUN="${START_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_BASE="${PROJECT_ROOT}/runs/reproduce/${RUN_TAG}"
LOG_DIR="${PROJECT_ROOT}/logs/reproduce"
mkdir -p "$RUN_BASE" "$LOG_DIR"

export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2048}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export N_COMPLETIONS="${N_COMPLETIONS:-32}"
export MAX_TOKENS="${MAX_TOKENS:-38912}"
export N_EXP_COMPLETIONS="${N_EXP_COMPLETIONS:-32}"
export THRESHOLD="${THRESHOLD:-0.8}"
export EMB_MODEL="${EMB_MODEL:-/data/ppnm/models/all-MiniLM-L6-v2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export START_INDEX="${START_INDEX:-0}"

vllm_extra_args() {
  if [ -n "${MAX_MODEL_LEN:-}" ]; then
    printf -- '--max-model-len %s' "$MAX_MODEL_LEN"
  fi
}
VLLM_EXTRA="$(vllm_extra_args)"

_has_files() {
  local dir="$1"
  compgen -G "${dir}/*" >/dev/null 2>&1
}

rse_pass1_iter() {
  local iter="$1"
  local ver_dir="$2"
  if [ ! -d "$ver_dir" ]; then
    echo "  iter${iter}  N/A  (missing dir: $ver_dir)"
    return 0
  fi
  python eval/calculate_pass_at_k_from_completions.py \
    --verification_dir "$ver_dir" \
    --k_values 1 \
    --output_file "${ver_dir}/pass_at_1.json" \
    --max_reference 32 \
    --tokenizer_path "$MODEL_NAME" >/dev/null
  local PASS1
  PASS1="$(python - "$ver_dir" <<'PY'
import json, sys
ver_dir = sys.argv[1]
with open(ver_dir + "/pass_at_1.json", "r", encoding="utf-8") as f:
    m = json.load(f)
v = m.get("pass_at_k", {}).get("pass@1", None)
print("N/A" if v is None else f"{v*100:.2f}%")
PY
)"
  echo "  iter${iter}  ${PASS1}  (${ver_dir}/pass_at_1.json)"
}

_run_one_pipeline() {
  local run_id="$1"
  local out_prefix="${RUN_BASE}/run${run_id}"
  local s1="${out_prefix}_step1/results"
  local s2="${out_prefix}_step2/results"
  local s2d="${out_prefix}_step2/results_dedup"
  local s3="${out_prefix}_step3/results"
  local s4="${out_prefix}_step4/results"
  local s4d="${out_prefix}_step4/results_dedup"
  local s5="${out_prefix}_step5/results"
  local s6="${out_prefix}_step6/results"
  local s6d="${out_prefix}_step6/results_dedup"
  local s7="${out_prefix}_step7/results"

  echo ""
  echo ">>>>>>>>>> Run ${run_id}/2 (resume) <<<<<<<<<<"

  if _has_files "$s7"; then
    echo "[skip] Run ${run_id} already complete (step7 results exist)"
    echo "---------- Pass@1 (iter0..3) ----------"
    rse_pass1_iter 0 "$s1"
    rse_pass1_iter 1 "$s3"
    rse_pass1_iter 2 "$s5"
    rse_pass1_iter 3 "$s7"
    echo "========== RSE Pipeline done: ${out_prefix} =========="
    return 0
  fi

  if _has_files "$s1"; then
    echo "[skip] Run ${run_id}: step1 already done ($s1)"
  else
    echo "---------- Step 1: Baseline Sampling ----------"
    mkdir -p "$s1"
    # shellcheck disable=SC2086
    python code/standard_sampling.py \
      --model "$MODEL_NAME" \
      --input "$QUESTION_FILE" \
      --output "$s1" \
      --n-completions "$N_COMPLETIONS" \
      --batch-size "$BATCH_SIZE" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" \
      --start-idx "$START_INDEX" \
      --end-idx "$END_INDEX" \
      $VLLM_EXTRA
  fi

  if _has_files "$s2"; then
    echo "[skip] Run ${run_id}: step2 already done ($s2)"
  else
    echo "---------- Step 2: Experience Distillation ----------"
    mkdir -p "$s2"
    # shellcheck disable=SC2086
    python code/experience_distillation.py \
      --model "$MODEL_NAME" \
      --question-file "$QUESTION_FILE" \
      --answer-dir "$s1" \
      --output-dir "$s2" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" \
      --n-samples 1 \
      --start-idx "$START_INDEX" \
      --end-idx "$END_INDEX" \
      $VLLM_EXTRA
  fi

  if _has_files "$s2d"; then
    echo "[skip] Run ${run_id}: step2 dedup already done ($s2d)"
  else
    echo "---------- Step 2.5: Experience Deduplication ----------"
    mkdir -p "$s2d" "${out_prefix}_step2/results_dedup_debug"
    python code/experience_dedup.py \
      --experience-dir "$s2" \
      --output-dir "$s2d" \
      --debug-dir "${out_prefix}_step2/results_dedup_debug" \
      --model-path "$EMB_MODEL" \
      --threshold "$THRESHOLD" \
      --keep-order
  fi

  if _has_files "$s3"; then
    echo "[skip] Run ${run_id}: step3 already done ($s3)"
  else
    echo "---------- Step 3: Experience-Guided Search (iter1) ----------"
    mkdir -p "$s3"
    # shellcheck disable=SC2086
    python code/experience_guided_search.py \
      --model "$MODEL_NAME" \
      --input "$QUESTION_FILE" \
      --experience-dir "$s2d" \
      --output "$s3" \
      --n-experience-completions "$N_EXP_COMPLETIONS" \
      --n-completions 32 \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" \
      --start-idx "$START_INDEX" \
      --end-idx "$END_INDEX" \
      $VLLM_EXTRA
  fi

  if _has_files "$s4"; then
    echo "[skip] Run ${run_id}: step4 already done ($s4)"
  else
    echo "---------- Step 4: Experience Distillation ----------"
    mkdir -p "$s4"
    # shellcheck disable=SC2086
    python code/experience_distillation.py \
      --model "$MODEL_NAME" \
      --question-file "$QUESTION_FILE" \
      --answer-dir "$s3" \
      --output-dir "$s4" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" \
      --n-samples 1 \
      --start-idx "$START_INDEX" \
      --end-idx "$END_INDEX" \
      $VLLM_EXTRA
  fi

  if _has_files "$s4d"; then
    echo "[skip] Run ${run_id}: step4 dedup already done ($s4d)"
  else
    mkdir -p "$s4d" "${out_prefix}_step4/results_dedup_debug"
    python code/experience_dedup.py \
      --experience-dir "$s4" \
      --previous-experience-dir "$s2d" \
      --output-dir "$s4d" \
      --debug-dir "${out_prefix}_step4/results_dedup_debug" \
      --model-path "$EMB_MODEL" \
      --threshold "$THRESHOLD" \
      --keep-order
  fi

  if _has_files "$s5"; then
    echo "[skip] Run ${run_id}: step5 already done ($s5)"
  else
    echo "---------- Step 5: Experience-Guided Search (iter2) ----------"
    mkdir -p "$s5"
    # shellcheck disable=SC2086
    python code/experience_guided_search.py \
      --model "$MODEL_NAME" \
      --input "$QUESTION_FILE" \
      --experience-dir "$s4d" \
      --output "$s5" \
      --n-experience-completions "$N_EXP_COMPLETIONS" \
      --n-completions 32 \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" \
      --start-idx "$START_INDEX" \
      --end-idx "$END_INDEX" \
      $VLLM_EXTRA
  fi

  if _has_files "$s6"; then
    echo "[skip] Run ${run_id}: step6 already done ($s6)"
  else
    echo "---------- Step 6: Experience Distillation ----------"
    mkdir -p "$s6"
    # shellcheck disable=SC2086
    python code/experience_distillation.py \
      --model "$MODEL_NAME" \
      --question-file "$QUESTION_FILE" \
      --answer-dir "$s5" \
      --output-dir "$s6" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" \
      --n-samples 1 \
      --start-idx "$START_INDEX" \
      --end-idx "$END_INDEX" \
      $VLLM_EXTRA
  fi

  if _has_files "$s6d"; then
    echo "[skip] Run ${run_id}: step6 dedup already done ($s6d)"
  else
    mkdir -p "$s6d" "${out_prefix}_step6/results_dedup_debug"
    python code/experience_dedup.py \
      --experience-dir "$s6" \
      --previous-experience-dir "$s4d" \
      --output-dir "$s6d" \
      --debug-dir "${out_prefix}_step6/results_dedup_debug" \
      --model-path "$EMB_MODEL" \
      --threshold "$THRESHOLD" \
      --keep-order
  fi

  if _has_files "$s7"; then
    echo "[skip] Run ${run_id}: step7 already done ($s7)"
  else
    echo "---------- Step 7: Experience-Guided Search (iter3) ----------"
    mkdir -p "$s7"
    # shellcheck disable=SC2086
    python code/experience_guided_search.py \
      --model "$MODEL_NAME" \
      --input "$QUESTION_FILE" \
      --experience-dir "$s6d" \
      --output "$s7" \
      --n-experience-completions "$N_EXP_COMPLETIONS" \
      --n-completions 32 \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" \
      --start-idx "$START_INDEX" \
      --end-idx "$END_INDEX" \
      $VLLM_EXTRA
  fi

  echo "---------- Pass@1 (iter0..3) ----------"
  rse_pass1_iter 0 "$s1"
  rse_pass1_iter 1 "$s3"
  rse_pass1_iter 2 "$s5"
  rse_pass1_iter 3 "$s7"
  echo "========== RSE Pipeline done: ${out_prefix} =========="
}

echo "============================================================"
echo "RSE Reproduce RESUME: ${RUN_TAG}"
echo "  GPU=${GPU_ID}  MODEL=${MODEL_NAME}"
echo "  DATASET=${DATASET_NAME}  END_INDEX=${END_INDEX}"
echo "  START_RUN=${START_RUN}  RUN_BASE=${RUN_BASE}"
echo "============================================================"

for run_id in $(seq "$START_RUN" 2); do
  _run_one_pipeline "$run_id"
done

python scripts/aggregate_rse_results.py \
  --run-base "$RUN_BASE" \
  --model "$MODEL_NAME" \
  --dataset "$DATASET_NAME" \
  --num-runs 3 \
  --output-csv "${RUN_BASE}/results.csv"

echo ""
echo "========== Resume done. CSV: ${RUN_BASE}/results.csv =========="
