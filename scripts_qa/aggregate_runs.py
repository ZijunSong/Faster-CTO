#!/usr/bin/env python3
"""汇总 NUM_RUNS 次实验的 iter Pass@1 mean±std。"""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--eval-iter", type=int, default=3)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    iter_tag = f"iter{args.eval_iter}"
    values = []
    per_run = []

    for run_id in range(1, args.num_runs + 1):
        summary_path = runs_root / f"run{run_id}_eval_summary.json"
        if not summary_path.exists():
            per_run.append({"run_id": run_id, "pass_at_1_pct": None, "error": "missing summary"})
            continue
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        entry = data.get(iter_tag, {})
        p = entry.get("pass_at_1")
        pct = entry.get("pass_at_1_pct")
        if p is None and pct is not None:
            p = pct / 100.0
        if p is not None:
            values.append(float(p))
            per_run.append({"run_id": run_id, "pass_at_1": p, "pass_at_1_pct": round(p * 100, 4)})
        else:
            per_run.append({"run_id": run_id, "pass_at_1_pct": None})

    if values:
        m = mean(values)
        s = pstdev(values) if len(values) > 1 else 0.0
        summary = {
            "dataset": args.dataset,
            "model_tag": args.model_tag,
            "eval_iter": iter_tag,
            "num_runs": args.num_runs,
            "n_valid": len(values),
            "mean_pass_at_1": round(m, 6),
            "std_pass_at_1": round(s, 6),
            "mean_pass_at_1_pct": round(m * 100, 4),
            "std_pass_at_1_pct": round(s * 100, 4),
            "values_pct": [round(v * 100, 4) for v in values],
            "per_run": per_run,
        }
        md = (
            f"# RSE {args.num_runs}-Run Summary\n\n"
            f"- **Dataset**: {args.dataset}\n"
            f"- **Model**: {args.model_tag}\n"
            f"- **Metric**: {iter_tag} Pass@1 (EM)\n\n"
            f"## Result\n\n"
            f"**{summary['mean_pass_at_1_pct']:.2f} ± {summary['std_pass_at_1_pct']:.2f}%** "
            f"(n={summary['n_valid']})\n\n"
            f"## Per-run\n\n"
        )
        for r in per_run:
            v = r.get("pass_at_1_pct")
            md += f"- run{r['run_id']}: {v if v is not None else 'N/A'}%\n"
        md += f"\n## Raw values (%)\n\n{summary['values_pct']}\n"
    else:
        summary = {
            "dataset": args.dataset,
            "model_tag": args.model_tag,
            "eval_iter": iter_tag,
            "num_runs": args.num_runs,
            "n_valid": 0,
            "mean_pass_at_1_pct": None,
            "std_pass_at_1_pct": None,
            "per_run": per_run,
        }
        md = f"# RSE Summary\n\nNo valid runs for {iter_tag}.\n"

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
