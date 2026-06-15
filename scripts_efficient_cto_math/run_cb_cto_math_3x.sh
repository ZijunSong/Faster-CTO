#!/bin/bash
# CB-CTO 完整流水线（step1→7），step3/5/7 使用 vLLM Chunked Candidate CTO

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EFFICIENT_CTO_METHOD=cb_cto
# shellcheck source=inc_efficient_cto_common.sh
source "$SCRIPT_DIR/inc_efficient_cto_common.sh"

_SYSTEM_PROMPT_ARGS=()
if [ -n "${SYSTEM_PROMPT:-}" ]; then
  _SYSTEM_PROMPT_ARGS=(--system-prompt "$SYSTEM_PROMPT")
fi

_VLLM_TP_ARGS=()
if [ "${TENSOR_PARALLEL_SIZE}" -gt 1 ]; then
  _VLLM_TP_ARGS=(--disable-custom-all-reduce)
fi

_TASK_ARGS=(--dataset "$DATASET")

_count_question_jsons() {
  find "$1" -maxdepth 1 -name '[0-9]*.json' 2>/dev/null | wc -l
}

_expected_questions() {
  local end="${END_INDEX:-30}"
  local start="${START_INDEX:-0}"
  echo $((end - start))
}

_run_cb_cto_search() {
  local exp_dir="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  python code/cb_cto_guided_search.py \
    --model "$MODEL_NAME" \
    --input "$QUESTION_FILE" \
    --experience-dir "$exp_dir" \
    --output "$out_dir" \
    --n-experience-completions "$N_EXP_COMPLETIONS" \
    --n-completions "$N_COMPLETIONS" \
    --alpha "$ALPHA" \
    --cb-cto-trigger "$CB_CTO_TRIGGER" \
    "${_CBCTO_THRESHOLD_ARGS[@]}" \
    --chunk-size "$CB_CTO_CHUNK_SIZE" \
    --chunk-candidates "$CB_CTO_CHUNK_CANDIDATES" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$CTO_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$CTO_MAX_MODEL_LEN" \
    --vllm-score-batch-size "$VLLM_SCORE_BATCH_SIZE" \
    --vllm-max-score-prompt-tokens "$VLLM_MAX_SCORE_PROMPT_TOKENS" \
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

_run_distill() {
  local answer_dir="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  python code/experience_distillation.py \
    --model "$MODEL_NAME" \
    --question-file "$QUESTION_FILE" \
    --answer-dir "$answer_dir" \
    --output-dir "$out_dir" \
    "${_TASK_ARGS[@]}" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$DISTILL_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$DISTILL_MAX_MODEL_LEN" \
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
  echo "========== CB-CTO(${CB_CTO_TRIGGER}) Run ${run_id}/${NUM_RUNS} | ${DATASET} × ${MODEL_TAG} =========="

  mkdir -p "$s1"
  _exp_q="$(_expected_questions)"
  _have_q="$(_count_question_jsons "$s1")"
  if [ "$_have_q" -ge "$_exp_q" ]; then
    echo "[skip] run${run_id} step1 (iter0): already have ${_have_q}/${_exp_q} questions"
  else
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

  _run_distill "$s1" "$s2"

  mkdir -p "$s2d" "${out_base}_step2/results_dedup_debug"
  python code/experience_dedup.py \
    --experience-dir "$s2" \
    --output-dir "$s2d" \
    --debug-dir "${out_base}_step2/results_dedup_debug" \
    --model-path "$EMB_MODEL" \
    --threshold "$THRESHOLD" \
    --keep-order

  _run_cb_cto_search "$s2d" "$s3"
  _run_distill "$s3" "$s4"

  mkdir -p "$s4d" "${out_base}_step4/results_dedup_debug"
  python code/experience_dedup.py \
    --experience-dir "$s4" \
    --previous-experience-dir "$s2d" \
    --output-dir "$s4d" \
    --debug-dir "${out_base}_step4/results_dedup_debug" \
    --model-path "$EMB_MODEL" \
    --threshold "$THRESHOLD" \
    --keep-order

  _run_cb_cto_search "$s4d" "$s5"
  _run_distill "$s5" "$s6"

  mkdir -p "$s6d" "${out_base}_step6/results_dedup_debug"
  python code/experience_dedup.py \
    --experience-dir "$s6" \
    --previous-experience-dir "$s4d" \
    --output-dir "$s6d" \
    --debug-dir "${out_base}_step6/results_dedup_debug" \
    --model-path "$EMB_MODEL" \
    --threshold "$THRESHOLD" \
    --keep-order

  _run_cb_cto_search "$s6d" "$s7"
}

_eval_one_run() {
  local run_id="$1"
  local out_base="${RUNS_ROOT}/run${run_id}"
  local summary_json="${out_base}_eval_summary.json"

  python scripts_cto_math/eval_math_iters.py \
    --tokenizer-path "$MODEL_PATH" \
    --out-base "$out_base" \
    --output-json "$summary_json" \
    --max-reference 32
}

for run_id in $(seq 1 "$NUM_RUNS"); do
  _run_one_pipeline "$run_id"
  _eval_one_run "$run_id"
done

SUMMARY_JSON="${RUNS_ROOT}/mean_std_summary.json"
SUMMARY_MD="${RUNS_ROOT}/mean_std_summary.md"

python scripts_efficient_cto_math/aggregate_efficient_cto_runs.py \
  --runs-root "$RUNS_ROOT" \
  --num-runs "$NUM_RUNS" \
  --eval-iter "$EVAL_ITER" \
  --dataset "$DATASET" \
  --model-tag "$MODEL_TAG" \
  --method "CB-CTO-${CB_CTO_TRIGGER}" \
  --output-json "$SUMMARY_JSON" \
  --output-md "$SUMMARY_MD"

python scripts_cto_qa/update_dataset_csv.py \
  --summary-json "$SUMMARY_JSON" \
  --csv-path "$RESULTS_CSV" \
  --model-name "$MODEL_NAME" \
  --method "CB-CTO-${CB_CTO_TRIGGER}"

echo ""
echo "========== CB-CTO(${CB_CTO_TRIGGER}) 全部 ${NUM_RUNS} 次运行完成 =========="
cat "$SUMMARY_MD"
