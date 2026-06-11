#!/bin/bash
# 断点续跑 CTO QA 流水线：已完成的 step 自动跳过，从未完成的 step 继续。
# 依赖 inc_cto_common.sh 已设置 MODEL/DATASET 等变量。
# 可选环境变量 START_RUN（默认 1）、RUNS_ROOT（指向已有结果目录）。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=inc_cto_common.sh
source "$SCRIPT_DIR/inc_cto_common.sh"

START_RUN="${START_RUN:-1}"

_SYSTEM_PROMPT_ARGS=()
if [ -n "${SYSTEM_PROMPT:-}" ]; then
  _SYSTEM_PROMPT_ARGS=(--system-prompt "$SYSTEM_PROMPT")
fi

_VLLM_TP_ARGS=()
if [ "${TENSOR_PARALLEL_SIZE}" -gt 1 ]; then
  _VLLM_TP_ARGS=(--disable-custom-all-reduce)
fi

_TASK_ARGS=(--dataset "$DATASET")

_has_files() {
  local dir="$1"
  compgen -G "${dir}/*" >/dev/null 2>&1
}

_run_cto_search() {
  local exp_dir="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  python code/cto_guided_search.py \
    --model "$MODEL_NAME" \
    --input "$QUESTION_FILE" \
    --experience-dir "$exp_dir" \
    --output "$out_dir" \
    --n-experience-completions "$N_EXP_COMPLETIONS" \
    --n-completions "$N_COMPLETIONS" \
    --alpha "$ALPHA" \
    --plausibility-top-k "$PLAUSIBILITY_TOP_K" \
    --backend vllm \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$CTO_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$CTO_MAX_MODEL_LEN" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --start-idx "$START_INDEX" \
    --end-idx "$END_INDEX" \
    "${_TASK_ARGS[@]}" \
    "${CTO_RETRIEVAL_ARGS[@]}" \
    "${_VLLM_TP_ARGS[@]}"
}

_run_one_pipeline() {
  local run_id="$1"
  local out_base="${RUNS_ROOT}/run${run_id}"
  local s1="${out_base}_step1/results"
  local s2="${out_base}_step2/results"
  local s2d="${out_base}_step2/results_dedup"
  local s3="${out_base}_step3/results"
  local s4="${out_base}_step4/results"
  local s4d="${out_base}_step4/results_dedup"
  local s5="${out_base}_step5/results"
  local s6="${out_base}_step6/results"
  local s6d="${out_base}_step6/results_dedup"
  local s7="${out_base}_step7/results"

  echo ""
  echo "========== CTO Run ${run_id}/${NUM_RUNS} | ${DATASET} × ${MODEL_TAG} (resume) =========="

  if _has_files "$s7" && [ -f "${out_base}_eval_summary.json" ]; then
    echo "[skip] Run ${run_id} already complete (step7 + eval_summary)"
    return 0
  fi

  if _has_files "$s1"; then
    echo "[skip] Run ${run_id}: step1 already done ($s1)"
  else
    mkdir -p "$s1"
    python code/standard_sampling.py \
      --model "$MODEL_NAME" \
      --input "$QUESTION_FILE" \
      --output "$s1" \
      "${_TASK_ARGS[@]}" \
      --n-completions "$N_COMPLETIONS" \
      --batch-size "$BATCH_SIZE" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" \
      --start-idx "$START_INDEX" \
      --end-idx "$END_INDEX" \
      "${_SYSTEM_PROMPT_ARGS[@]}" \
      "${_VLLM_TP_ARGS[@]}"
  fi

  if _has_files "$s2"; then
    echo "[skip] Run ${run_id}: step2 already done ($s2)"
  else
    mkdir -p "$s2"
    python code/experience_distillation.py \
      --model "$MODEL_NAME" \
      --question-file "$QUESTION_FILE" \
      --answer-dir "$s1" \
      --output-dir "$s2" \
      "${_TASK_ARGS[@]}" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$DISTILL_MAX_TOKENS" \
      --n-samples 1 \
      --experience_judge_mode "$EXPERIENCE_JUDGE_MODE" \
      --start-idx "$START_INDEX" \
      --end-idx "$END_INDEX" \
      "${_VLLM_TP_ARGS[@]}"
  fi

  if _has_files "$s2d"; then
    echo "[skip] Run ${run_id}: step2 dedup already done ($s2d)"
  else
    mkdir -p "$s2d" "${out_base}_step2/results_dedup_debug"
    python code/experience_dedup.py \
      --experience-dir "$s2" \
      --output-dir "$s2d" \
      --debug-dir "${out_base}_step2/results_dedup_debug" \
      --model-path "$EMB_MODEL" \
      --threshold "$THRESHOLD" \
      --keep-order
  fi

  if _has_files "$s3"; then
    echo "[skip] Run ${run_id}: step3 already done ($s3)"
  else
    _run_cto_search "$s2d" "$s3"
  fi

  if _has_files "$s4"; then
    echo "[skip] Run ${run_id}: step4 already done ($s4)"
  else
    mkdir -p "$s4"
    python code/experience_distillation.py \
      --model "$MODEL_NAME" \
      --question-file "$QUESTION_FILE" \
      --answer-dir "$s3" \
      --output-dir "$s4" \
      "${_TASK_ARGS[@]}" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$DISTILL_MAX_TOKENS" \
      --n-samples 1 \
      --experience_judge_mode "$EXPERIENCE_JUDGE_MODE" \
      --start-idx "$START_INDEX" \
      --end-idx "$END_INDEX" \
      "${_VLLM_TP_ARGS[@]}"
  fi

  if _has_files "$s4d"; then
    echo "[skip] Run ${run_id}: step4 dedup already done ($s4d)"
  else
    mkdir -p "$s4d" "${out_base}_step4/results_dedup_debug"
    python code/experience_dedup.py \
      --experience-dir "$s4" \
      --previous-experience-dir "$s2d" \
      --output-dir "$s4d" \
      --debug-dir "${out_base}_step4/results_dedup_debug" \
      --model-path "$EMB_MODEL" \
      --threshold "$THRESHOLD" \
      --keep-order
  fi

  if _has_files "$s5"; then
    echo "[skip] Run ${run_id}: step5 already done ($s5)"
  else
    _run_cto_search "$s4d" "$s5"
  fi

  if _has_files "$s6"; then
    echo "[skip] Run ${run_id}: step6 already done ($s6)"
  else
    mkdir -p "$s6"
    python code/experience_distillation.py \
      --model "$MODEL_NAME" \
      --question-file "$QUESTION_FILE" \
      --answer-dir "$s5" \
      --output-dir "$s6" \
      "${_TASK_ARGS[@]}" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$DISTILL_MAX_TOKENS" \
      --n-samples 1 \
      --experience_judge_mode "$EXPERIENCE_JUDGE_MODE" \
      --start-idx "$START_INDEX" \
      --end-idx "$END_INDEX" \
      "${_VLLM_TP_ARGS[@]}"
  fi

  if _has_files "$s6d"; then
    echo "[skip] Run ${run_id}: step6 dedup already done ($s6d)"
  else
    mkdir -p "$s6d" "${out_base}_step6/results_dedup_debug"
    python code/experience_dedup.py \
      --experience-dir "$s6" \
      --previous-experience-dir "$s4d" \
      --output-dir "$s6d" \
      --debug-dir "${out_base}_step6/results_dedup_debug" \
      --model-path "$EMB_MODEL" \
      --threshold "$THRESHOLD" \
      --keep-order
  fi

  if _has_files "$s7"; then
    echo "[skip] Run ${run_id}: step7 already done ($s7)"
  else
    _run_cto_search "$s6d" "$s7"
  fi
}

_eval_one_run() {
  local run_id="$1"
  local out_base="${RUNS_ROOT}/run${run_id}"
  local summary_json="${out_base}_eval_summary.json"

  python scripts_qa/eval_qa_iters.py \
    --tokenizer-path "$MODEL_PATH" \
    --out-base "$out_base" \
    --output-json "$summary_json" \
    --max-reference 32
}

for run_id in $(seq "$START_RUN" "$NUM_RUNS"); do
  _run_one_pipeline "$run_id"
  _eval_one_run "$run_id"
done

SUMMARY_JSON="${RUNS_ROOT}/mean_std_summary.json"
SUMMARY_MD="${RUNS_ROOT}/mean_std_summary.md"

python scripts_qa/aggregate_runs.py \
  --runs-root "$RUNS_ROOT" \
  --num-runs "$NUM_RUNS" \
  --eval-iter "$EVAL_ITER" \
  --dataset "$DATASET" \
  --model-tag "$MODEL_TAG" \
  --task-type "cto_qa" \
  --output-json "$SUMMARY_JSON" \
  --output-md "$SUMMARY_MD"

python scripts_cto_qa/update_dataset_csv.py \
  --summary-json "$SUMMARY_JSON" \
  --csv-path "$RESULTS_CSV" \
  --model-name "$MODEL_NAME" \
  --method CTO

python scripts_qa/format_wide_results.py \
  --project-root "$PROJECT_ROOT" \
  --dataset "$DATASET" \
  --runs-root "$RUNS_ROOT"

echo ""
echo "========== CTO 全部 ${NUM_RUNS} 次运行完成（resume） =========="
echo "本模型汇总: $SUMMARY_JSON"
echo "数据集 CSV: $RESULTS_CSV"
cat "$SUMMARY_MD"
