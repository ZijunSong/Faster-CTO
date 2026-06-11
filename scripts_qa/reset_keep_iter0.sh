#!/bin/bash
# 保留 run{1,2,3}_step1（iter0），删除其余 step 与汇总文件。
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <runs_root> [<runs_root> ...]"
  exit 1
fi

for runs_root in "$@"; do
  if [ ! -d "$runs_root" ]; then
    echo "[skip] not found: $runs_root"
    continue
  fi
  echo "[clean] $runs_root"
  for run_id in 1 2 3; do
    for step in step2 step3 step4 step5 step6 step7; do
      target="${runs_root}/run${run_id}_${step}"
      if [ -e "$target" ]; then
        rm -rf "$target"
        echo "  removed run${run_id}_${step}"
      fi
    done
    eval_json="${runs_root}/run${run_id}_eval_summary.json"
    if [ -f "$eval_json" ]; then
      rm -f "$eval_json"
      echo "  removed run${run_id}_eval_summary.json"
    fi
  done
  rm -f "${runs_root}/mean_std_summary.json" "${runs_root}/mean_std_summary.md"
done
