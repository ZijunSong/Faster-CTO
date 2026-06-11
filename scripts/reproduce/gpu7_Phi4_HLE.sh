#!/bin/bash
# GPU 7: Phi-4-reasoning on HLE_math_text, 3 runs, single GPU
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export TMPDIR=/data/tmp
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=/data/ppnm/.cache
mkdir -p "$XDG_CACHE_HOME"

export GPU_ID=7
export MODEL_NAME="/data/ppnm/models/Phi-4-reasoning"
export DATASET_NAME="HLE_math_text"
export QUESTION_FILE="${PROJECT_ROOT}/data/HLE_MATH_text_100_sample_subset.jsonl"
export END_INDEX=100
export RUN_TAG="HLE_math_text_Phi-4-reasoning"
export MAX_MODEL_LEN=32768
export MAX_TOKENS=24576

bash "${PROJECT_ROOT}/scripts/common/run_reproduce_3x.sh"
