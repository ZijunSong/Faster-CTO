#!/bin/bash
# RSE | HotpotQA | Phi-4-Reasoning | 双卡 GPU2-3 TP=2 | 3 runs mean±std
# Usage: bash scripts_qa/GPU2-3_HotpotQA_Phi4_Reasoning.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=2,3
export TENSOR_PARALLEL_SIZE=2
export GPU_GROUP=GPU2-3
export DATASET=HotpotQA
export MODEL_TAG=Phi4_Reasoning
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_rse_3x.sh" "${1:-0}" "${2:-}"
