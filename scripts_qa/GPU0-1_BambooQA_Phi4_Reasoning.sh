#!/bin/bash
# RSE | BambooQA | Phi-4-Reasoning | 双卡 GPU0-1 TP=2 | 3 runs mean±std
# Usage: bash scripts_qa/GPU0-1_BambooQA_Phi4_Reasoning.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=0,1
export TENSOR_PARALLEL_SIZE=2
export GPU_GROUP=GPU0-1
export DATASET=BambooQA
export MODEL_TAG=Phi4_Reasoning
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_rse_3x.sh" "${1:-0}" "${2:-}"
