#!/bin/bash
# CTO | HotpotQA | Qwen3-30B-A3B-Thinking-2507 | 四卡 GPU4-7 TP=4 | 3 runs mean±std
# Usage: bash scripts_cto_qa/GPU4-7_HotpotQA_Qwen3_30B_A3B_Thinking.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TENSOR_PARALLEL_SIZE=4
export GPU_GROUP=GPU4-7
export DATASET=HotpotQA
export MODEL_TAG=Qwen3_30B_A3B_Thinking
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_cto_3x.sh" "${1:-0}" "${2:-}"
