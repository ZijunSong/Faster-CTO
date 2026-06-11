#!/bin/bash
# RSE | BambooQA | Qwen3-30B-A3B-Thinking-2507 | 四卡 GPU0-3 TP=4 | 3 runs mean±std
# Usage: bash scripts_qa/GPU0-3_BambooQA_Qwen3_30B_A3B_Thinking.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TENSOR_PARALLEL_SIZE=4
export GPU_GROUP=GPU0-3
export DATASET=BambooQA
export MODEL_TAG=Qwen3_30B_A3B_Thinking
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_rse_3x.sh" "${1:-0}" "${2:-}"
