#!/bin/bash
# CTO | HMMT24 | Phi-4-Reasoning | 双卡 GPU4-5 TP=2 | 3 runs mean±std
set -e
export CUDA_VISIBLE_DEVICES=4,5
export TENSOR_PARALLEL_SIZE=2
export GPU_GROUP=GPU4-5
export DATASET=HMMT24
export MODEL_TAG=Phi4_Reasoning
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_cto_math_3x.sh" "${1:-0}" "${2:-}"
