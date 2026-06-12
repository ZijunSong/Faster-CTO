#!/bin/bash
# B-CTO trigger=rollout_disagreement | HMMT24 | Qwen3-4B-Thinking-2507 | 3 runs mean±std
# 风险信号：argmax(logits_pos) 与 stochastic sample 不一致时启用 negative branch
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPU_GROUP="${GPU_GROUP:-GPU0}"
export DATASET=HMMT24
export MODEL_TAG=Qwen3_4B_Thinking
export B_CTO_TRIGGER=rollout_disagreement
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/scripts_b_cto_math/run_b_cto_math_3x.sh" "${1:-0}" "${2:-}"
