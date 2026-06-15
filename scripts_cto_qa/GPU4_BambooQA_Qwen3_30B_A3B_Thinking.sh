#!/bin/bash
# CTO | BambooQA | Qwen3-30B-A3B-Thinking-2507 | 单卡 GPU4 | 3 runs mean±std
# llm_judge + 同题 embedding_rerank（默认见 inc_cto_common.sh）
# Usage: bash scripts_cto_qa/GPU4_BambooQA_Qwen3_30B_A3B_Thinking.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=4
export GPU_GROUP=GPU4
export MODELS_ROOT="${MODELS_ROOT:-/data/ppnm/models}"
export DATASET=BambooQA
export MODEL_TAG=Qwen3_30B_A3B_Thinking
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_cto_3x.sh" "${1:-0}" "${2:-}"
