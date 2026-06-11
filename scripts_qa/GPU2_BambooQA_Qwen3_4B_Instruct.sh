#!/bin/bash
# RSE | BambooQA | Qwen3-4B-Instruct-2507 | 单卡 GPU2 | 3 runs mean±std
# Usage: bash scripts_qa/GPU2_BambooQA_Qwen3_4B_Instruct.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=2
export DATASET=BambooQA
export MODEL_TAG=Qwen3_4B_Instruct
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_rse_3x.sh" "${1:-0}" "${2:-}"
