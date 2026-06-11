#!/bin/bash
# Run RSE pipeline 3 times and aggregate Pass@1 into one CSV.
# Required env: GPU_ID, MODEL_NAME, DATASET_NAME, QUESTION_FILE, END_INDEX, RUN_TAG
# Optional env: MAX_MODEL_LEN, MAX_TOKENS, TENSOR_PARALLEL_SIZE
set -e

: "${GPU_ID:?GPU_ID is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${DATASET_NAME:?DATASET_NAME is required}"
: "${QUESTION_FILE:?QUESTION_FILE is required}"
: "${END_INDEX:?END_INDEX is required}"
: "${RUN_TAG:?RUN_TAG is required}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_BASE="${PROJECT_ROOT}/runs/reproduce/${RUN_TAG}"
LOG_DIR="${PROJECT_ROOT}/logs/reproduce"
mkdir -p "$RUN_BASE" "$LOG_DIR"

echo "============================================================"
echo "RSE Reproduce: ${RUN_TAG}"
echo "  GPU=${GPU_ID}  MODEL=${MODEL_NAME}"
echo "  DATASET=${DATASET_NAME}  END_INDEX=${END_INDEX}"
echo "  RUN_BASE=${RUN_BASE}"
echo "============================================================"

for run_id in 0 1 2; do
  echo ""
  echo ">>>>>>>>>> Run ${run_id}/2 <<<<<<<<<<"
  export OUT_PREFIX="${RUN_BASE}/run${run_id}"
  bash "${SCRIPT_DIR}/rse_full_pipeline.sh" 0 "${END_INDEX}"
done

python scripts/aggregate_rse_results.py \
  --run-base "$RUN_BASE" \
  --model "$MODEL_NAME" \
  --dataset "$DATASET_NAME" \
  --num-runs 3 \
  --output-csv "${RUN_BASE}/results.csv"

echo ""
echo "========== All 3 runs done. CSV: ${RUN_BASE}/results.csv =========="
