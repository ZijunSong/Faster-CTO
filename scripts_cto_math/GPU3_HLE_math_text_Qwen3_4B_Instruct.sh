#!/bin/bash
# CTO | HLE_math_text | Qwen3-4B-Instruct-2507 | 单卡 GPU3 | 3 runs mean±std
set -e
export CUDA_VISIBLE_DEVICES=3
export GPU_GROUP=GPU3
export DATASET=HLE_math_text
export MODEL_TAG=Qwen3_4B_Instruct
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_cto_math_3x.sh" "${1:-0}" "${2:-}"
