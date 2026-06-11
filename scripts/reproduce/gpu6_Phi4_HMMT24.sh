#!/bin/bash
# GPU 6: Phi-4-reasoning on HMMT24, 3 runs, single GPU
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export TMPDIR=/data/tmp
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=/data/ppnm/.cache
mkdir -p "$XDG_CACHE_HOME"

export GPU_ID=6
export MODEL_NAME="/data/ppnm/models/Phi-4-reasoning"
export DATASET_NAME="HMMT24"
export QUESTION_FILE="${PROJECT_ROOT}/data/HMMT_24.jsonl"
export END_INDEX=30
export RUN_TAG="HMMT24_Phi-4-reasoning"
export MAX_MODEL_LEN=32768
export MAX_TOKENS=24576

bash "${PROJECT_ROOT}/scripts/common/run_reproduce_3x.sh"
