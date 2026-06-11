#!/bin/bash
# 断点续跑 | RSE HotpotQA | Qwen3-4B-Thinking | GPU6（结果写入原 GPU1 目录）
# Usage: bash scripts_qa/GPU6_HotpotQA_Qwen3_4B_Thinking_resume.sh [start] [end]
set -e
export CUDA_VISIBLE_DEVICES=6
export GPU_GROUP=GPU1
export DATASET=HotpotQA
export MODEL_TAG=Qwen3_4B_Thinking
export RUNS_ROOT="/home/test/test12/songzijun/RSE/runs_qa/HotpotQA/Qwen3_4B_Thinking_GPU1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/resume_rse_3x.sh" "${1:-0}" "${2:-}"
