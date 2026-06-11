#!/bin/bash
# 断点续跑 | CTO HotpotQA | Qwen3-4B-Thinking | GPU7（结果写入原 GPU5 目录）
# Usage: bash scripts_cto_qa/GPU7_HotpotQA_Qwen3_4B_Thinking_resume.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=7
export DATASET=HotpotQA
export MODEL_TAG=Qwen3_4B_Thinking
export RUNS_ROOT="/home/test/test12/songzijun/RSE/runs_cto_qa/HotpotQA/Qwen3_4B_Thinking_GPU5"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/resume_cto_3x.sh" "${1:-0}" "${2:-}"
