#!/bin/bash
# RSE | BambooQA | Qwen3-4B-Thinking-2507 | 单卡 GPU0 | 3 runs mean±std
# Usage: bash scripts_qa/GPU0_BambooQA_Qwen3_4B_Thinking.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=0
export DATASET=BambooQA
export MODEL_TAG=Qwen3_4B_Thinking
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_rse_3x.sh" "${1:-0}" "${2:-}"
