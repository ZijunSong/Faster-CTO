#!/bin/bash
# CTO | HotpotQA | Qwen3-4B-Thinking-2507 | 单卡 GPU5 | 3 runs mean±std
# Usage: bash scripts_cto_qa/GPU5_HotpotQA_Qwen3_4B_Thinking.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=5
export DATASET=HotpotQA
export MODEL_TAG=Qwen3_4B_Thinking
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_cto_3x.sh" "${1:-0}" "${2:-}"
