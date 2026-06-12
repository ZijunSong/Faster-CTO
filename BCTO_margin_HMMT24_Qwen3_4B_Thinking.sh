#!/bin/bash
# B-CTO trigger=margin | HMMT24 | Qwen3-4B-Thinking-2507 | 3 runs mean±std
# 风险信号：top-1 与 top-2 logit 差值 < threshold（默认 1.0）时启用 negative branch
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPU_GROUP="${GPU_GROUP:-GPU0}"
export DATASET=HMMT24
export MODEL_TAG=Qwen3_4B_Thinking
export B_CTO_TRIGGER=margin
export B_CTO_THRESHOLD="${B_CTO_THRESHOLD:-1.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/scripts_b_cto_math/run_b_cto_math_3x.sh" "${1:-0}" "${2:-}"
