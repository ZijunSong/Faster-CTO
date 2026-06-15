#!/bin/bash
# HMMT24 | Qwen3-4B-Thinking | CB-CTO | 单卡 GPU0 | 3 runs
# 继承 reproduce run0/1/2 的 iter0 → 本流水线 run1/2/3 step1，然后从 step2 开始
#
# nohup 运行（推荐）:
#   cd /data/ppnm/RSE
#   nohup bash nohup_hmmt24_CBCTO_GPU0.sh > logs/efficient_cto_math/nohup_cbcto_gpu0.log 2>&1 &
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

# ========== 实验配置 ==========
export GPU_ID=0
export CUDA_VISIBLE_DEVICES=0
export GPU_GROUP=GPU0
export DATASET=HMMT24
export MODEL_TAG=Qwen3_4B_Thinking
export NUM_RUNS=3
export EVAL_ITER=3

export MODELS_ROOT="${MODELS_ROOT:-/data/ppnm/models}"
export CB_CTO_TRIGGER="${CB_CTO_TRIGGER:-entropy}"
export CB_CTO_THRESHOLD="${CB_CTO_THRESHOLD:-}"
export CB_CTO_CHUNK_SIZE="${CB_CTO_CHUNK_SIZE:-16}"
export CB_CTO_CHUNK_CANDIDATES="${CB_CTO_CHUNK_CANDIDATES:-4}"

export RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs_cb_cto_math/HMMT24/${MODEL_TAG}_cb_cto_${CB_CTO_TRIGGER}_${GPU_GROUP}}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/efficient_cto_math}"

echo "============================================================"
echo "CB-CTO HMMT24 | GPU${GPU_ID} | trigger=${CB_CTO_TRIGGER} | ${NUM_RUNS} runs"
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
exec bash "${PROJECT_ROOT}/scripts_efficient_cto_math/run_cb_cto_math_3x.sh" 0 30
