#!/bin/bash
# 将 runs/reproduce/HMMT24_Qwen3-4B-Thinking-2507 的 iter0(step1) 复制到新流水线目录
# reproduce 使用 run0/1/2；efficient CTO 流水线使用 run1/2/3
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_ROOT="${SRC_ROOT:-${PROJECT_ROOT}/runs/reproduce/HMMT24_Qwen3-4B-Thinking-2507}"
DST_ROOT="${1:?usage: reuse_hmmt24_iter0_step1.sh <dst_runs_root>}"

if [ ! -d "$SRC_ROOT" ]; then
  echo "ERROR: 源目录不存在: $SRC_ROOT"
  exit 1
fi

mkdir -p "$DST_ROOT"

for src_run in 0 1 2; do
  dst_run=$((src_run + 1))
  src="${SRC_ROOT}/run${src_run}_step1/results"
  dst="${DST_ROOT}/run${dst_run}_step1/results"
  if [ ! -d "$src" ]; then
    echo "WARN: 跳过缺失源 run${src_run}_step1"
    continue
  fi
  mkdir -p "$dst"
  # 只复制题目级 json，跳过 pass_at_1.json 等汇总文件
  find "$src" -maxdepth 1 -name '[0-9]*.json' -exec cp -f {} "$dst/" \;
  n=$(find "$dst" -maxdepth 1 -name '[0-9]*.json' | wc -l)
  echo "  run${dst_run}_step1 <= reproduce run${src_run}_step1 : ${n} questions"
done

echo "Done. Reused 3× iter0 (reproduce run0/1/2 → pipeline run1/2/3) into ${DST_ROOT}"
