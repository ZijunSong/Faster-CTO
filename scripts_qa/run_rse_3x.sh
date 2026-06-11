#!/bin/bash
# 完整 RSE 流水线（step1→7），重复 NUM_RUNS 次并写入数据集级 CSV
# 依赖 inc_common.sh 已设置 MODEL/DATASET 等变量

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=inc_common.sh
source "$SCRIPT_DIR/inc_common.sh"

_SYSTEM_PROMPT_ARGS=()
if [ -n "${SYSTEM_PROMPT:-}" ]; then
  _SYSTEM_PROMPT_ARGS=(--system-prompt "$SYSTEM_PROMPT")
fi

_VLLM_TP_ARGS=()
if [ "${TENSOR_PARALLEL_SIZE}" -gt 1 ]; then
  _VLLM_TP_ARGS=(--disable-custom-all-reduce)
fi

_TASK_ARGS=(--dataset "$DATASET")

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
  echo "========== Run ${run_id}/${NUM_RUNS} | ${DATASET} × ${MODEL_TAG} =========="

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
    --max-model-len "$DISTILL_MAX_MODEL_LEN" \
    --max-tokens "$DISTILL_MAX_TOKENS" \
    --n-samples 1 \
    --start-idx "$START_INDEX" \
    --end-idx "$END_INDEX" \
    "${_VLLM_TP_ARGS[@]}"

  mkdir -p "$s2d" "${out_base}_step2/results_dedup_debug"
  python code/experience_dedup.py \
    --experience-dir "$s2" \
    --output-dir "$s2d" \
    --debug-dir "${out_base}_step2/results_dedup_debug" \
    --model-path "$EMB_MODEL" \
    --threshold "$THRESHOLD" \
    --keep-order

  mkdir -p "$s3"
  python code/experience_guided_search.py \
    --model "$MODEL_NAME" \
    --input "$QUESTION_FILE" \
    --experience-dir "$s2d" \
    --output "$s3" \
    "${_TASK_ARGS[@]}" \
    --n-experience-completions "$N_EXP_COMPLETIONS" \
    --n-completions "$N_COMPLETIONS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --start-idx "$START_INDEX" \
    --end-idx "$END_INDEX" \
    "${_SYSTEM_PROMPT_ARGS[@]}" \
    "${_VLLM_TP_ARGS[@]}"

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
    --max-model-len "$DISTILL_MAX_MODEL_LEN" \
    --max-tokens "$DISTILL_MAX_TOKENS" \
    --n-samples 1 \
    --start-idx "$START_INDEX" \
    --end-idx "$END_INDEX" \
    "${_VLLM_TP_ARGS[@]}"

  mkdir -p "$s4d" "${out_base}_step4/results_dedup_debug"
  python code/experience_dedup.py \
    --experience-dir "$s4" \
    --previous-experience-dir "$s2d" \
    --output-dir "$s4d" \
    --debug-dir "${out_base}_step4/results_dedup_debug" \
    --model-path "$EMB_MODEL" \
    --threshold "$THRESHOLD" \
    --keep-order

  mkdir -p "$s5"
  python code/experience_guided_search.py \
    --model "$MODEL_NAME" \
    --input "$QUESTION_FILE" \
    --experience-dir "$s4d" \
    --output "$s5" \
    "${_TASK_ARGS[@]}" \
    --n-experience-completions "$N_EXP_COMPLETIONS" \
    --n-completions "$N_COMPLETIONS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --start-idx "$START_INDEX" \
    --end-idx "$END_INDEX" \
    "${_SYSTEM_PROMPT_ARGS[@]}" \
    "${_VLLM_TP_ARGS[@]}"

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
    --max-model-len "$DISTILL_MAX_MODEL_LEN" \
    --max-tokens "$DISTILL_MAX_TOKENS" \
    --n-samples 1 \
    --start-idx "$START_INDEX" \
    --end-idx "$END_INDEX" \
    "${_VLLM_TP_ARGS[@]}"

  mkdir -p "$s6d" "${out_base}_step6/results_dedup_debug"
  python code/experience_dedup.py \
    --experience-dir "$s6" \
    --previous-experience-dir "$s4d" \
    --output-dir "$s6d" \
    --debug-dir "${out_base}_step6/results_dedup_debug" \
    --model-path "$EMB_MODEL" \
    --threshold "$THRESHOLD" \
    --keep-order

  mkdir -p "$s7"
  python code/experience_guided_search.py \
    --model "$MODEL_NAME" \
    --input "$QUESTION_FILE" \
    --experience-dir "$s6d" \
    --output "$s7" \
    "${_TASK_ARGS[@]}" \
    --n-experience-completions "$N_EXP_COMPLETIONS" \
    --n-completions "$N_COMPLETIONS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --start-idx "$START_INDEX" \
    --end-idx "$END_INDEX" \
    "${_SYSTEM_PROMPT_ARGS[@]}" \
    "${_VLLM_TP_ARGS[@]}"
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

for run_id in $(seq 1 "$NUM_RUNS"); do
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
  --output-json "$SUMMARY_JSON" \
  --output-md "$SUMMARY_MD"

python scripts_qa/update_dataset_csv.py \
  --summary-json "$SUMMARY_JSON" \
  --csv-path "$RESULTS_CSV" \
  --model-name "$MODEL_NAME"

echo ""
echo "========== 全部 ${NUM_RUNS} 次运行完成 =========="
echo "本模型汇总: $SUMMARY_JSON"
echo "数据集 CSV: $RESULTS_CSV"
cat "$SUMMARY_MD"
