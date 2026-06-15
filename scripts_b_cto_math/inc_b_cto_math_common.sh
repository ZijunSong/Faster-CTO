#!/bin/bash
# B-CTO 数学实验公共配置
# 需设置: CUDA_VISIBLE_DEVICES, DATASET, MODEL_TAG, B_CTO_TRIGGER

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

export MODELS_ROOT="${MODELS_ROOT:-/home/test/test12/models}"
export EMB_MODEL="${EMB_MODEL:-${MODELS_ROOT}/all-MiniLM-L6-v2}"
export RETRIEVAL_RERANK_MODEL="${RETRIEVAL_RERANK_MODEL:-${MODELS_ROOT}/cross-encoder-ms-marco-MiniLM-L-6-v2}"

export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2048}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export N_COMPLETIONS="${N_COMPLETIONS:-32}"
export N_EXP_COMPLETIONS="${N_EXP_COMPLETIONS:-32}"
export THRESHOLD="${THRESHOLD:-0.85}"

export NUM_RUNS="${NUM_RUNS:-3}"
export EVAL_ITER="${EVAL_ITER:-3}"

export CTO_EXPERIENCE_RETRIEVAL="${CTO_EXPERIENCE_RETRIEVAL:-embedding_rerank}"
export CTO_RERANK_POOL_MULT="${CTO_RERANK_POOL_MULT:-4}"
export EXPERIENCE_JUDGE_MODE="${EXPERIENCE_JUDGE_MODE:-llm_judge}"

export ALPHA="${ALPHA:-0.7}"
export PLAUSIBILITY_TOP_K="${PLAUSIBILITY_TOP_K:-5}"

# B-CTO trigger thresholds (mode-specific defaults in Python if unset)
export B_CTO_TRIGGER="${B_CTO_TRIGGER:?B_CTO_TRIGGER must be set}"
export B_CTO_THRESHOLD="${B_CTO_THRESHOLD:-}"

export DISTILL_MAX_MODEL_LEN="${DISTILL_MAX_MODEL_LEN:-100000}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.90}"
export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-38912}"

export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export TMPDIR="${TMPDIR:-/tmp/b_cto_math_${B_CTO_TRIGGER}_${GPU_GROUP:-gpu${CUDA_VISIBLE_DEVICES}}_$$}"
mkdir -p "$TMPDIR"
export RAY_TMPDIR="${RAY_TMPDIR:-$TMPDIR}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.cache}"
mkdir -p "$XDG_CACHE_HOME"

if [ -z "${CC:-}" ] && [ -x /opt/rh/devtoolset-11/root/usr/bin/gcc ]; then
  export CC=/opt/rh/devtoolset-11/root/usr/bin/gcc
  export CXX=/opt/rh/devtoolset-11/root/usr/bin/g++
fi

START_INDEX="${1:-0}"
END_INDEX="${2:-}"

case "${MODEL_TAG}" in
  Qwen3_4B_Thinking)
    export MODEL_NAME="${MODEL_NAME:-${MODELS_ROOT}/Qwen3-4B-Thinking-2507}"
    export MAX_TOKENS="${MAX_TOKENS:-38912}"
    export B_CTO_DEVICE="${B_CTO_DEVICE:-cuda:0}"
    export B_CTO_DEVICE_MAP="${B_CTO_DEVICE_MAP:-}"
    export B_CTO_DTYPE="${B_CTO_DTYPE:-bfloat16}"
    ;;
  Qwen3_30B_Thinking|Qwen3_30B_A3B_Thinking)
    export MODEL_NAME="${MODEL_NAME:-${MODELS_ROOT}/Qwen3-30B-A3B-Thinking-2507}"
    export MAX_TOKENS="${MAX_TOKENS:-38912}"
    export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-8192}"
    export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.95}"
    export B_CTO_DEVICE="${B_CTO_DEVICE:-cuda:0}"
    export B_CTO_DEVICE_MAP="${B_CTO_DEVICE_MAP:-auto}"
    export B_CTO_DTYPE="${B_CTO_DTYPE:-bfloat16}"
    ;;
  *)
    echo "ERROR: 未知 MODEL_TAG=${MODEL_TAG}（支持 Qwen3_4B_Thinking / Qwen3_30B_Thinking）"
    exit 1
    ;;
esac
export MODEL_PATH="${MODEL_NAME}"

case "${DATASET}" in
  HMMT24)
    export QUESTION_FILE="${QUESTION_FILE:-${PROJECT_ROOT}/data/HMMT_24.jsonl}"
    END_INDEX="${END_INDEX:-30}"
    ;;
  *)
    echo "ERROR: B-CTO scripts currently support DATASET=HMMT24 only (got ${DATASET})"
    exit 1
    ;;
esac

if [ ! -f "$QUESTION_FILE" ]; then
  echo "ERROR: 数据文件不存在: $QUESTION_FILE"
  exit 1
fi

export SYSTEM_PROMPT="${SYSTEM_PROMPT:-Please reason step by step, and put your final answer within \\boxed{}.}"
export GPU_GROUP="${GPU_GROUP:-GPU${CUDA_VISIBLE_DEVICES}}"
export RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs_b_cto_math/${DATASET}/${MODEL_TAG}_${B_CTO_TRIGGER}_${GPU_GROUP}}"
export RESULTS_CSV="${RESULTS_CSV:-${PROJECT_ROOT}/results_b_cto_math/${DATASET}_results.csv}"
mkdir -p "$RUNS_ROOT" "$(dirname "$RESULTS_CSV")"

if [ "${CTO_EXPERIENCE_RETRIEVAL}" = "first" ]; then
  export CTO_RETRIEVAL_ARGS=(--experience-retrieval first)
else
  export CTO_RETRIEVAL_ARGS=(
    --experience-retrieval "${CTO_EXPERIENCE_RETRIEVAL}"
    --retrieval-embedding-model "$EMB_MODEL"
    --retrieval-rerank-model "$RETRIEVAL_RERANK_MODEL"
    --retrieval-rerank-pool-mult "${CTO_RERANK_POOL_MULT}"
  )
fi

_BCTO_THRESHOLD_ARGS=()
if [ -n "${B_CTO_THRESHOLD}" ]; then
  _BCTO_THRESHOLD_ARGS=(--b-cto-threshold "$B_CTO_THRESHOLD")
fi

echo "======================================================="
echo "B-CTO Math Experiment"
echo "  DATASET=${DATASET}  MODEL_TAG=${MODEL_TAG}"
echo "  TRIGGER=${B_CTO_TRIGGER}  THRESHOLD=${B_CTO_THRESHOLD:-<default>}"
echo "  GPU=${CUDA_VISIBLE_DEVICES}  GPU_GROUP=${GPU_GROUP}"
echo "  MODEL=${MODEL_NAME}"
echo "  DATA=${QUESTION_FILE}"
echo "  RANGE=[${START_INDEX}, ${END_INDEX})  NUM_RUNS=${NUM_RUNS}"
echo "  OUT=${RUNS_ROOT}"
echo "  CSV=${RESULTS_CSV}"
echo "======================================================="
