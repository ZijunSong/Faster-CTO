#!/bin/bash
# CTO-Rescore (sparse negative reranking) | HMMT24 | Qwen3-4B-Thinking-2507
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPU_GROUP="${GPU_GROUP:-GPU0}"
export DATASET=HMMT24
export MODEL_TAG=Qwen3_4B_Thinking
export CTO_RESCORE_N_GENERATE="${CTO_RESCORE_N_GENERATE:-32}"
export CTO_RESCORE_M="${CTO_RESCORE_M:-8}"
export CTO_RESCORE_SELECTION="${CTO_RESCORE_SELECTION:-top_pos}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/scripts_efficient_cto_math/run_cto_rescore_math_3x.sh" "${1:-0}" "${2:-}"
