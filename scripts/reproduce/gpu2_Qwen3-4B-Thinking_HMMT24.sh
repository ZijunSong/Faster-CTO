#!/bin/bash
# GPU 2: Qwen3-4B-Thinking-2507 on HMMT24, 3 runs, single GPU
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export GPU_ID=2
export MODEL_NAME="/data/ppnm/models/Qwen3-4B-Thinking-2507"
export DATASET_NAME="HMMT24"
export QUESTION_FILE="${PROJECT_ROOT}/data/HMMT_24.jsonl"
export END_INDEX=30
export RUN_TAG="HMMT24_Qwen3-4B-Thinking-2507"

bash "${PROJECT_ROOT}/scripts/common/run_reproduce_3x.sh"
