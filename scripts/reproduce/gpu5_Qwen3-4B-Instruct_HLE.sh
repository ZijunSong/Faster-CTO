#!/bin/bash
# GPU 5: Qwen3-4B-Instruct-2507 on HLE_math_text, 3 runs, single GPU
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export GPU_ID=5
export MODEL_NAME="/data/ppnm/models/Qwen3-4B-Instruct-2507"
export DATASET_NAME="HLE_math_text"
export QUESTION_FILE="${PROJECT_ROOT}/data/HLE_MATH_text_100_sample_subset.jsonl"
export END_INDEX=100
export RUN_TAG="HLE_math_text_Qwen3-4B-Instruct-2507"

bash "${PROJECT_ROOT}/scripts/common/run_reproduce_3x.sh"
