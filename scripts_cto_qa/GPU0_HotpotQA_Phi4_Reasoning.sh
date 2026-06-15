#!/bin/bash
# CTO | HotpotQA | Phi-4-Reasoning | 单卡 GPU0 | 3 runs mean±std
# llm_judge + 同题 embedding_rerank（默认见 inc_cto_common.sh）
# Usage: bash scripts_cto_qa/GPU0_HotpotQA_Phi4_Reasoning.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=0
export GPU_GROUP=GPU0
export MODELS_ROOT="${MODELS_ROOT:-/data/ppnm/models}"
export DATASET=HotpotQA
export MODEL_TAG=Phi4_Reasoning
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_cto_3x.sh" "${1:-0}" "${2:-}"
