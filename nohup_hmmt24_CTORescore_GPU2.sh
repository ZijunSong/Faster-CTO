#!/bin/bash
# HMMT24 | Qwen3-4B-Thinking | CTO-Rescore | 单卡 GPU2 | 3 runs
# 继承 reproduce run0/1/2 的 iter0 → 本流水线 run1/2/3 step1，然后从 step2 开始
#
# nohup 运行（推荐）:
#   cd /data/ppnm/RSE
#   nohup bash nohup_hmmt24_CTORescore_GPU2.sh > logs/efficient_cto_math/nohup_cto_rescore_gpu2.log 2>&1 &
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

# ========== 实验配置 ==========
export GPU_ID=2
export CUDA_VISIBLE_DEVICES=2
export GPU_GROUP=GPU2
export DATASET=HMMT24
export MODEL_TAG=Qwen3_4B_Thinking
export NUM_RUNS=3
export EVAL_ITER=3

export MODELS_ROOT="${MODELS_ROOT:-/data/ppnm/models}"
export CTO_RESCORE_N_GENERATE="${CTO_RESCORE_N_GENERATE:-32}"
export CTO_RESCORE_M="${CTO_RESCORE_M:-8}"
export CTO_RESCORE_SELECTION="${CTO_RESCORE_SELECTION:-top_pos}"
export CTO_RESCORE_RISK_THRESHOLD="${CTO_RESCORE_RISK_THRESHOLD:-0.85}"

export RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs_cto_rescore_math/HMMT24/${MODEL_TAG}_cto_rescore_${GPU_GROUP}}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/efficient_cto_math}"

echo "============================================================"
echo "CTO-Rescore HMMT24 | GPU${GPU_ID} | N=${CTO_RESCORE_N_GENERATE} M=${CTO_RESCORE_M} | ${NUM_RUNS} runs"
echo "  START: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  OUT:   ${RUNS_ROOT}"
echo "============================================================"

# ========== 前置检查 ==========
# shellcheck source=scripts_efficient_cto_math/preflight_hmmt24_qwen3_4b.sh
source "${PROJECT_ROOT}/scripts_efficient_cto_math/preflight_hmmt24_qwen3_4b.sh"

# ========== 继承 3× iter0 ==========
echo "[reuse] 复制 reproduce iter0 → ${RUNS_ROOT} (run0/1/2 → run1/2/3) ..."
bash "${PROJECT_ROOT}/scripts_efficient_cto_math/reuse_hmmt24_iter0_step1.sh" "$RUNS_ROOT"

# ========== 3× 完整流水线 (step2→7 ×3 + eval) ==========
exec bash "${PROJECT_ROOT}/scripts_efficient_cto_math/run_cto_rescore_math_3x.sh" 0 30
