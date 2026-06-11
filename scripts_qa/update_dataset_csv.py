#!/usr/bin/env python3
"""将单模型 3 次运行汇总写入数据集级 CSV（支持并行写入锁）。"""
import argparse
import csv
import fcntl
import json
from datetime import datetime
from pathlib import Path

COLUMNS = [
    "dataset",
    "model_tag",
    "model_path",
    "eval_iter",
    "metric",
    "mean_pct",
    "std_pct",
    "mean",
    "std",
    "n_valid",
    "num_runs",
    "run1_pct",
    "run2_pct",
    "run3_pct",
    "updated_at",
]


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLUMNS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    csv_path = Path(args.csv_path)
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    per_run = summary.get("per_run", [])
    run_pcts = [""] * 3
    for entry in per_run:
        rid = entry.get("run_id")
        if isinstance(rid, int) and 1 <= rid <= 3:
            v = entry.get("pass_at_1_pct")
            run_pcts[rid - 1] = "" if v is None else f"{v:.4f}"

    new_row = {
        "dataset": summary.get("dataset", ""),
        "model_tag": summary.get("model_tag", ""),
        "model_path": args.model_name,
        "eval_iter": summary.get("eval_iter", "iter3"),
        "metric": "Pass@1_EM",
        "mean_pct": "" if summary.get("mean_pass_at_1_pct") is None else f"{summary['mean_pass_at_1_pct']:.4f}",
        "std_pct": "" if summary.get("std_pass_at_1_pct") is None else f"{summary['std_pass_at_1_pct']:.4f}",
        "mean": "" if summary.get("mean_pass_at_1") is None else f"{summary['mean_pass_at_1']:.6f}",
        "std": "" if summary.get("std_pass_at_1") is None else f"{summary['std_pass_at_1']:.6f}",
        "n_valid": summary.get("n_valid", ""),
        "num_runs": summary.get("num_runs", 3),
        "run1_pct": run_pcts[0],
        "run2_pct": run_pcts[1],
        "run3_pct": run_pcts[2],
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(lock_path, "w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        rows = load_rows(csv_path)
        key = (new_row["dataset"], new_row["model_tag"])
        updated = False
        for i, row in enumerate(rows):
            if (row.get("dataset"), row.get("model_tag")) == key:
                rows[i] = new_row
                updated = True
                break
        if not updated:
            rows.append(new_row)
        write_rows(csv_path, rows)
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    print(f"Updated CSV: {csv_path}")


if __name__ == "__main__":
    main()
