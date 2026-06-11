#!/usr/bin/env python3
"""Aggregate RSE Pass@1 results from multiple runs into a CSV with mean±std."""

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

ITER_STEPS = [1, 3, 5, 7]
ITER_LABELS = ["iter0", "iter1", "iter2", "iter3"]


def load_pass_at_1(results_dir: Path) -> Optional[float]:
    path = results_dir / "pass_at_1.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pass_at_k", {}).get("pass@1")


def ensure_eval(results_dir: Path, model_name: str) -> Optional[float]:
    if not results_dir.is_dir():
        return None
    pass_path = results_dir / "pass_at_1.json"
    if not pass_path.exists():
        import subprocess
        subprocess.run(
            [
                "python",
                "eval/calculate_pass_at_k_from_completions.py",
                "--verification_dir",
                str(results_dir),
                "--k_values",
                "1",
                "--output_file",
                str(pass_path),
                "--max_reference",
                "32",
                "--tokenizer_path",
                model_name,
            ],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    return load_pass_at_1(results_dir)


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def fmt_mean_std(values: List[float]) -> str:
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean * 100:.2f}±{std * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate RSE runs into CSV")
    parser.add_argument("--run-base", required=True, help="Base dir containing run0, run1, ...")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    run_base = Path(args.run_base)
    fieldnames = ["run_id", "model", "dataset"] + ITER_LABELS
    rows: List[Dict[str, str]] = []
    iter_values: Dict[str, List[float]] = {label: [] for label in ITER_LABELS}

    for run_id in range(args.num_runs):
        run_prefix = run_base / f"run{run_id}"
        row = {
            "run_id": str(run_id),
            "model": args.model,
            "dataset": args.dataset,
        }
        for label, step in zip(ITER_LABELS, ITER_STEPS):
            results_dir = Path(f"{run_prefix}_step{step}/results")
            value = ensure_eval(results_dir, args.model)
            row[label] = fmt_pct(value)
            if value is not None:
                iter_values[label].append(value)
        rows.append(row)

    summary_row = {
        "run_id": "mean±std",
        "model": args.model,
        "dataset": args.dataset,
    }
    for label in ITER_LABELS:
        summary_row[label] = fmt_mean_std(iter_values[label])
    rows.append(summary_row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_csv}")
    for label in ITER_LABELS:
        print(f"  {label}: {summary_row[label]}")


if __name__ == "__main__":
    main()
