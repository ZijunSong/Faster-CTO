#!/bin/bash
# CTO | HLE_math_text | Phi-4-Reasoning | 双卡 GPU6-7 TP=2 | 3 runs mean±std
set -e
export CUDA_VISIBLE_DEVICES=6,7
export TENSOR_PARALLEL_SIZE=2
export GPU_GROUP=GPU6-7
export DATASET=HLE_math_text
export MODEL_TAG=Phi4_Reasoning
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_cto_math_3x.sh" "${1:-0}" "${2:-}"
