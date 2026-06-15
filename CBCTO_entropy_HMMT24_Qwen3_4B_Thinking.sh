#!/bin/bash
# CB-CTO trigger=entropy | HMMT24 | Qwen3-4B-Thinking-2507 | 3 runs mean±std
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPU_GROUP="${GPU_GROUP:-GPU0}"
export DATASET=HMMT24
export MODEL_TAG=Qwen3_4B_Thinking
export CB_CTO_TRIGGER=entropy
export CB_CTO_THRESHOLD="${CB_CTO_THRESHOLD:-0.85}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/scripts_efficient_cto_math/run_cb_cto_math_3x.sh" "${1:-0}" "${2:-}"
