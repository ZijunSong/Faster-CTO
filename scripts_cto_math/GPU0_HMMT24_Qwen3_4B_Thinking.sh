#!/bin/bash
# CTO | HMMT24 | Qwen3-4B-Thinking-2507 | 单卡 GPU0 | 3 runs mean±std
set -e
export CUDA_VISIBLE_DEVICES=0
export GPU_GROUP=GPU0
export DATASET=HMMT24
export MODEL_TAG=Qwen3_4B_Thinking
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_cto_math_3x.sh" "${1:-0}" "${2:-}"
