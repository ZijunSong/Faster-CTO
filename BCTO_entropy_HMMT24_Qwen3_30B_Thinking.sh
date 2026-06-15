#!/bin/bash
# B-CTO trigger=entropy | HMMT24 | Qwen3-30B-A3B-Thinking-2507 | 单卡 GPU2 | 3 runs mean±std
# 风险信号：positive branch softmax 归一化熵 > threshold（默认 0.85）时启用 negative branch
# Usage: bash BCTO_entropy_HMMT24_Qwen3_30B_Thinking.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=2
export GPU_GROUP=GPU2
export MODELS_ROOT="${MODELS_ROOT:-/data/ppnm/models}"
export DATASET=HMMT24
export MODEL_TAG=Qwen3_30B_Thinking
export B_CTO_TRIGGER=entropy
export B_CTO_THRESHOLD="${B_CTO_THRESHOLD:-0.85}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/scripts_b_cto_math/run_b_cto_math_3x.sh" "${1:-0}" "${2:-}"
