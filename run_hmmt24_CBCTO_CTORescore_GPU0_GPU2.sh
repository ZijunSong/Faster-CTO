#!/bin/bash
# HMMT24 | Qwen3-4B-Thinking | 一键并行测评
#   GPU0 → CB-CTO (默认 entropy trigger)
#   GPU2 → CTO-Rescore (sparse negative reranking)
#
# Usage:
#   bash run_hmmt24_CBCTO_CTORescore_GPU0_GPU2.sh
#   REUSE_ITER0=0 bash run_hmmt24_CBCTO_CTORescore_GPU0_GPU2.sh   # 不复用 step1，从头采样
#   CB_CTO_TRIGGER=margin bash run_hmmt24_CBCTO_CTORescore_GPU0_GPU2.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MODELS_ROOT="${MODELS_ROOT:-/data/ppnm/models}"
export PROJECT_ROOT="$SCRIPT_DIR"
export DATASET=HMMT24
export MODEL_TAG=Qwen3_4B_Thinking
export REUSE_ITER0="${REUSE_ITER0:-1}"

export CB_CTO_TRIGGER="${CB_CTO_TRIGGER:-entropy}"
export CB_CTO_THRESHOLD="${CB_CTO_THRESHOLD:-}"
export CB_CTO_CHUNK_SIZE="${CB_CTO_CHUNK_SIZE:-16}"
export CB_CTO_CHUNK_CANDIDATES="${CB_CTO_CHUNK_CANDIDATES:-4}"

export CTO_RESCORE_N_GENERATE="${CTO_RESCORE_N_GENERATE:-32}"
export CTO_RESCORE_M="${CTO_RESCORE_M:-8}"
export CTO_RESCORE_SELECTION="${CTO_RESCORE_SELECTION:-top_pos}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/efficient_cto_math}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

CBCTO_GPU="${CBCTO_GPU:-0}"
RESCORE_GPU="${RESCORE_GPU:-2}"
CBCTO_GPU_GROUP="${CBCTO_GPU_GROUP:-GPU${CBCTO_GPU}}"
RESCORE_GPU_GROUP="${RESCORE_GPU_GROUP:-GPU${RESCORE_GPU}}"

CBCTO_RUNS_ROOT="${PROJECT_ROOT}/runs_cb_cto_math/HMMT24/${MODEL_TAG}_cb_cto_${CB_CTO_TRIGGER}_${CBCTO_GPU_GROUP}"
RESCORE_RUNS_ROOT="${PROJECT_ROOT}/runs_cto_rescore_math/HMMT24/${MODEL_TAG}_cto_rescore_${RESCORE_GPU_GROUP}"

echo "============================================================"
echo "HMMT24 Efficient CTO 并行启动"
echo "  模型: Qwen3-4B-Thinking-2507"
echo "  GPU${CBCTO_GPU}: CB-CTO trigger=${CB_CTO_TRIGGER}"
echo "  GPU${RESCORE_GPU}: CTO-Rescore N=${CTO_RESCORE_N_GENERATE} M=${CTO_RESCORE_M}"
echo "  REUSE_ITER0=${REUSE_ITER0}"
echo "  CB-CTO OUT: ${CBCTO_RUNS_ROOT}"
echo "  Rescore OUT: ${RESCORE_RUNS_ROOT}"
echo "  LOG_DIR: ${LOG_DIR}"
echo "============================================================"

if [ "$REUSE_ITER0" = "1" ]; then
  echo "[reuse] CB-CTO step1 ..."
  bash "${SCRIPT_DIR}/scripts_efficient_cto_math/reuse_hmmt24_iter0_step1.sh" "$CBCTO_RUNS_ROOT"
  echo "[reuse] CTO-Rescore step1 ..."
  bash "${SCRIPT_DIR}/scripts_efficient_cto_math/reuse_hmmt24_iter0_step1.sh" "$RESCORE_RUNS_ROOT"
else
  echo "[reuse] 已禁用，将从 step1 采样开始完整运行"
fi

CBCTO_LOG="${LOG_DIR}/gpu${CBCTO_GPU}_cbcto_${CB_CTO_TRIGGER}_${TS}.log"
RESCORE_LOG="${LOG_DIR}/gpu${RESCORE_GPU}_cto_rescore_${TS}.log"

nohup env \
  CUDA_VISIBLE_DEVICES="$CBCTO_GPU" \
  GPU_GROUP="$CBCTO_GPU_GROUP" \
  MODELS_ROOT="$MODELS_ROOT" \
  PROJECT_ROOT="$PROJECT_ROOT" \
  DATASET="$DATASET" \
  MODEL_TAG="$MODEL_TAG" \
  CB_CTO_TRIGGER="$CB_CTO_TRIGGER" \
  CB_CTO_THRESHOLD="$CB_CTO_THRESHOLD" \
  CB_CTO_CHUNK_SIZE="$CB_CTO_CHUNK_SIZE" \
  CB_CTO_CHUNK_CANDIDATES="$CB_CTO_CHUNK_CANDIDATES" \
  RUNS_ROOT="$CBCTO_RUNS_ROOT" \
  bash "${SCRIPT_DIR}/scripts_efficient_cto_math/run_cb_cto_math_3x.sh" \
  >"$CBCTO_LOG" 2>&1 &
CBCTO_PID=$!

nohup env \
  CUDA_VISIBLE_DEVICES="$RESCORE_GPU" \
  GPU_GROUP="$RESCORE_GPU_GROUP" \
  MODELS_ROOT="$MODELS_ROOT" \
  PROJECT_ROOT="$PROJECT_ROOT" \
  DATASET="$DATASET" \
  MODEL_TAG="$MODEL_TAG" \
  CTO_RESCORE_N_GENERATE="$CTO_RESCORE_N_GENERATE" \
  CTO_RESCORE_M="$CTO_RESCORE_M" \
  CTO_RESCORE_SELECTION="$CTO_RESCORE_SELECTION" \
  RUNS_ROOT="$RESCORE_RUNS_ROOT" \
  bash "${SCRIPT_DIR}/scripts_efficient_cto_math/run_cto_rescore_math_3x.sh" \
  >"$RESCORE_LOG" 2>&1 &
RESCORE_PID=$!

echo ""
echo "已后台启动:"
echo "  CB-CTO     PID=${CBCTO_PID}  GPU=${CBCTO_GPU}  log=${CBCTO_LOG}"
echo "  CTO-Rescore PID=${RESCORE_PID}  GPU=${RESCORE_GPU}  log=${RESCORE_LOG}"
echo ""
echo "查看进度:"
echo "  tail -f ${CBCTO_LOG}"
echo "  tail -f ${RESCORE_LOG}"
echo "  watch -n30 nvidia-smi"
