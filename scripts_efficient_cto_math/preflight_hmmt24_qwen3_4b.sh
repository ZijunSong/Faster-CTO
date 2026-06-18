#!/bin/bash
# HMMT24 + Qwen3-4B-Thinking 实验前置检查（CTO-Rescore）
set -euo pipefail

_preflight_die() { echo "ERROR: $*" >&2; exit 1; }

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

export MODELS_ROOT="${MODELS_ROOT:-/data/ppnm/models}"
export CONDA_ENV="${CONDA_ENV:-cto}"
export CONDA_ROOT="${CONDA_ROOT:-/data/ppnm/miniconda3}"

if [ -z "${PYTHON_BIN:-}" ] && [ -x "${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python" ]; then
  export PYTHON_BIN="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
fi
export PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi
export PATH="$(dirname "$PYTHON_BIN"):${PATH}"
export DATASET="${DATASET:-HMMT24}"
export MODEL_TAG="${MODEL_TAG:-Qwen3_4B_Thinking}"
export NUM_RUNS="${NUM_RUNS:-3}"
export GPU_ID="${GPU_ID:?GPU_ID must be set}"

# ---------- Python / 依赖 ----------
command -v "$PYTHON_BIN" >/dev/null 2>&1 || _preflight_die "找不到 Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import vllm, transformers, torch, tqdm" \
  || _preflight_die "Python 依赖缺失，请安装 requirements.txt (vllm/transformers/torch/tqdm)"

# ---------- 路径 ----------
export MODEL_NAME="${MODEL_NAME:-${MODELS_ROOT}/Qwen3-4B-Thinking-2507}"
export EMB_MODEL="${EMB_MODEL:-${MODELS_ROOT}/all-MiniLM-L6-v2}"
export RETRIEVAL_RERANK_MODEL="${RETRIEVAL_RERANK_MODEL:-${MODELS_ROOT}/cross-encoder-ms-marco-MiniLM-L-6-v2}"
export QUESTION_FILE="${QUESTION_FILE:-${PROJECT_ROOT}/data/HMMT_24.jsonl}"
export ITER0_SRC="${ITER0_SRC:-${PROJECT_ROOT}/runs/reproduce/HMMT24_Qwen3-4B-Thinking-2507}"

[ -f "$MODEL_NAME/config.json" ] || _preflight_die "模型不存在: $MODEL_NAME"
[ -f "$QUESTION_FILE" ] || _preflight_die "数据不存在: $QUESTION_FILE"
[ -d "$ITER0_SRC" ] || _preflight_die "iter0 源目录不存在: $ITER0_SRC"
[ -d "$EMB_MODEL" ] || _preflight_die "嵌入模型不存在: $EMB_MODEL"
[ -d "$RETRIEVAL_RERANK_MODEL" ] || _preflight_die "Rerank 模型不存在: $RETRIEVAL_RERANK_MODEL"

HMMT24_N=$(wc -l <"$QUESTION_FILE")
[ "$HMMT24_N" -ge 1 ] || _preflight_die "HMMT24 数据为空"

for src_run in 0 1 2; do
  src="${ITER0_SRC}/run${src_run}_step1/results"
  [ -d "$src" ] || _preflight_die "缺少 reproduce run${src_run}_step1: $src"
  n=$(find "$src" -maxdepth 1 -name '[0-9]*.json' | wc -l)
  [ "$n" -ge "$HMMT24_N" ] || _preflight_die "run${src_run}_step1 仅 ${n}/${HMMT24_N} 题，iter0 不完整"
done

# ---------- GPU ----------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  _preflight_die "nvidia-smi 不可用"
fi
nvidia-smi -i "$GPU_ID" >/dev/null 2>&1 \
  || _preflight_die "GPU ${GPU_ID} 不可用（nvidia-smi -i ${GPU_ID} 失败）"

# ---------- 目录 ----------
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/efficient_cto_math}"
mkdir -p "$LOG_DIR" "${PROJECT_ROOT}/.cache"
export TMPDIR="${TMPDIR:-/tmp/efficient_cto_${GPU_ID}_$$}"
mkdir -p "$TMPDIR"
export RAY_TMPDIR="${RAY_TMPDIR:-$TMPDIR}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.cache}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -z "${CC:-}" ] && [ -x /opt/rh/devtoolset-11/root/usr/bin/gcc ]; then
  export CC=/opt/rh/devtoolset-11/root/usr/bin/gcc
  export CXX=/opt/rh/devtoolset-11/root/usr/bin/g++
fi

echo "[preflight] OK | GPU=${GPU_ID} | python=${PYTHON_BIN}"
echo "[preflight] model=${MODEL_NAME}"
echo "[preflight] HMMT24 questions=${HMMT24_N} | NUM_RUNS=${NUM_RUNS}"
echo "[preflight] iter0 source=${ITER0_SRC}"
