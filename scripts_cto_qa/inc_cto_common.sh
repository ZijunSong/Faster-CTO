#!/bin/bash
# CTO QA 实验公共配置（Qwen3-4B/30B × BambooQA / HotpotQA）
# 默认：LLM-Judge + 同题 embedding_rerank（bi-encoder + cross-encoder）
# 调用前需设置: CUDA_VISIBLE_DEVICES, DATASET, MODEL_TAG
# 多卡实验另设: TENSOR_PARALLEL_SIZE, GPU_GROUP

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

# CTO 默认：同题轨迹内 embedding + cross-encoder rerank
export CTO_EXPERIENCE_RETRIEVAL="${CTO_EXPERIENCE_RETRIEVAL:-embedding_rerank}"
export CTO_RERANK_POOL_MULT="${CTO_RERANK_POOL_MULT:-4}"
export EXPERIENCE_JUDGE_MODE="${EXPERIENCE_JUDGE_MODE:-llm_judge}"

# 与 RSE QA 对齐的 CTO 解码参数（4B）
export ALPHA="${ALPHA:-0.7}"
export PLAUSIBILITY_TOP_K="${PLAUSIBILITY_TOP_K:-5}"
export CTO_GPU_MEMORY_UTILIZATION="${CTO_GPU_MEMORY_UTILIZATION:-0.6}"
export CTO_MAX_MODEL_LEN="${CTO_MAX_MODEL_LEN:-100000}"
export DISTILL_MAX_MODEL_LEN="${DISTILL_MAX_MODEL_LEN:-100000}"
export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.90}"

export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export TMPDIR="${TMPDIR:-/tmp/cto_qa_${GPU_GROUP:-gpu${CUDA_VISIBLE_DEVICES}}_$$}"
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

if [ "${TENSOR_PARALLEL_SIZE}" -gt 1 ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _cvd_n=$(printf "%s" "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
  if [ "$_cvd_n" -lt "$TENSOR_PARALLEL_SIZE" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES 需至少 ${TENSOR_PARALLEL_SIZE} 张卡，当前 ${_cvd_n} 张 (${CUDA_VISIBLE_DEVICES})"
    exit 1
  fi
fi

case "${MODEL_TAG}" in
  Qwen3_4B_Thinking)
    export MODEL_NAME="${MODEL_NAME:-${MODELS_ROOT}/Qwen3-4B-Thinking-2507}"
    export MAX_TOKENS="${MAX_TOKENS:-38912}"
    export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-38912}"
    ;;
  Qwen3_4B_Instruct)
    export MODEL_NAME="${MODEL_NAME:-${MODELS_ROOT}/Qwen3-4B-Instruct-2507}"
    export MAX_TOKENS="${MAX_TOKENS:-38912}"
    export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-38912}"
    ;;
  Qwen3_30B_A3B_Thinking)
    export MODEL_NAME="${MODEL_NAME:-${MODELS_ROOT}/Qwen3-30B-A3B-Thinking-2507}"
    export MAX_TOKENS="${MAX_TOKENS:-38912}"
    export DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-8192}"
    export CTO_GPU_MEMORY_UTILIZATION="${CTO_GPU_MEMORY_UTILIZATION:-0.88}"
    export DISTILL_GPU_MEMORY_UTILIZATION="${DISTILL_GPU_MEMORY_UTILIZATION:-0.95}"
    ;;
  *)
    echo "ERROR: 未知 MODEL_TAG=${MODEL_TAG}"
    exit 1
    ;;
esac
export MODEL_PATH="${MODEL_NAME}"

case "${DATASET}" in
  BambooQA)
    export QUESTION_FILE="${QUESTION_FILE:-${PROJECT_ROOT}/data/BambooQA.jsonl}"
    END_INDEX="${END_INDEX:-125}"
    ;;
  HotpotQA)
    export QUESTION_FILE="${QUESTION_FILE:-${PROJECT_ROOT}/data/HotpotQA.jsonl}"
    END_INDEX="${END_INDEX:-2000}"
    ;;
  *)
    echo "ERROR: 未知 DATASET=${DATASET}"
    exit 1
    ;;
esac

if [ ! -f "$QUESTION_FILE" ]; then
  echo "ERROR: 数据文件不存在: $QUESTION_FILE"
  exit 1
fi

export SYSTEM_PROMPT="${SYSTEM_PROMPT:-Answer the question based on your knowledge. Reason step by step when needed, then give a concise final answer at the end in the format: Final answer: <your answer>.}"
export GPU_GROUP="${GPU_GROUP:-GPU${CUDA_VISIBLE_DEVICES}}"
export RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs_cto_qa/${DATASET}/${MODEL_TAG}_${GPU_GROUP}}"
export RESULTS_CSV="${RESULTS_CSV:-${PROJECT_ROOT}/results_cto_qa/${DATASET}_results.csv}"
mkdir -p "$RUNS_ROOT" "$(dirname "$RESULTS_CSV")"

if [ "${CTO_EXPERIENCE_RETRIEVAL}" = "first" ]; then
  export CTO_RETRIEVAL_ARGS=(
    --experience-retrieval first
  )
else
  export CTO_RETRIEVAL_ARGS=(
    --experience-retrieval "${CTO_EXPERIENCE_RETRIEVAL}"
    --retrieval-embedding-model "$EMB_MODEL"
    --retrieval-rerank-model "$RETRIEVAL_RERANK_MODEL"
    --retrieval-rerank-pool-mult "${CTO_RERANK_POOL_MULT}"
  )
fi

echo "======================================================="
echo "CTO QA Experiment"
echo "  DATASET=${DATASET}  MODEL_TAG=${MODEL_TAG}"
echo "  GPU=${CUDA_VISIBLE_DEVICES}  GPU_GROUP=${GPU_GROUP}  TP=${TENSOR_PARALLEL_SIZE}"
echo "  MODEL=${MODEL_NAME}"
echo "  DATA=${QUESTION_FILE}"
echo "  JUDGE=${EXPERIENCE_JUDGE_MODE}  RETRIEVAL=${CTO_EXPERIENCE_RETRIEVAL}"
echo "  RANGE=[${START_INDEX}, ${END_INDEX})  NUM_RUNS=${NUM_RUNS}"
echo "  OUT=${RUNS_ROOT}"
echo "  CSV=${RESULTS_CSV}"
echo "======================================================="
