#!/bin/bash
# CTO | BambooQA | Phi-4-Reasoning | 单卡 GPU5 | 3 runs mean±std
# llm_judge + 同题 embedding_rerank（默认见 inc_cto_common.sh）
# Usage: bash scripts_cto_qa/GPU5_BambooQA_Phi4_Reasoning.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=5
export GPU_GROUP=GPU5
export MODELS_ROOT="${MODELS_ROOT:-/data/ppnm/models}"
export DATASET=BambooQA
export MODEL_TAG=Phi4_Reasoning
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_cto_3x.sh" "${1:-0}" "${2:-}"
