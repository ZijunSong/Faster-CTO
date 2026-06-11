#!/bin/bash
# CTO | HLE_math_text | Qwen3-30B-A3B-Thinking-2507 | 四卡 GPU4-7 TP=4 | 3 runs mean±std
set -e
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TENSOR_PARALLEL_SIZE=4
export GPU_GROUP=GPU4-7
export DATASET=HLE_math_text
export MODEL_TAG=Qwen3_30B_A3B_Thinking
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_cto_math_3x.sh" "${1:-0}" "${2:-}"
