#!/bin/bash
# RSE | BambooQA | Phi-4-Reasoning | GPU0-3 分区
# Phi-4 GQA: num_key_value_heads=10，合法 TP 仅为 1/2/5/10，不能用 TP=4。
# 本脚本在 GPU0-1 上以 TP=2 运行，输出目录仍记为 Phi4_Reasoning_GPU0-3。
# Usage: bash scripts_qa/GPU0-3_BambooQA_Phi4_Reasoning.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=0,1
export TENSOR_PARALLEL_SIZE=2
export GPU_GROUP=GPU0-3
export DATASET=BambooQA
export MODEL_TAG=Phi4_Reasoning
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_rse_3x.sh" "${1:-0}" "${2:-}"
