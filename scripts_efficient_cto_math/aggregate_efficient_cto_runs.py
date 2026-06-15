#!/usr/bin/env python3
"""汇总 CB-CTO / CTO-Rescore 多次运行的 Pass@1 与效率指标。"""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def _collect_efficiency(runs_root: Path, num_runs: int, eval_iter: int) -> dict:
    step = {0: "step1", 1: "step3", 2: "step5", 3: "step7"}[eval_iter]
    processed = []
    generated = []
    neg_frac = []

    for run_id in range(1, num_runs + 1):
        res_dir = runs_root / f"run{run_id}_{step}" / "results"
        if not res_dir.is_dir():
            continue
        run_processed = []
        run_generated = []
        run_neg = []
        for fp in sorted(res_dir.glob("*.json")):
            try:
                with open(fp, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            comp = (
                data.get("cb_cto_compute")
                or data.get("cto_rescore_compute")
                or data.get("cto_compute")
                or {}
            )
            if "processed_tokens_per_rollout_mean" in comp:
                run_processed.append(float(comp["processed_tokens_per_rollout_mean"]))
            if "generated_tokens_mean" in comp:
                run_generated.append(float(comp["generated_tokens_mean"]))
            elif data.get("completions"):
                toks = [c.get("tokens", 0) for c in data["completions"]]
                if toks:
                    run_generated.append(float(mean(toks)))
            if "neg_branch_fraction" in comp:
                run_neg.append(float(comp["neg_branch_fraction"]))
        if run_processed:
            processed.append(mean(run_processed))
        if run_generated:
            generated.append(mean(run_generated))
        if run_neg:
            neg_frac.append(mean(run_neg))

    def _ms(xs):
        if not xs:
            return {"mean": None, "std": None, "values": []}
        m = mean(xs)
        s = pstdev(xs) if len(xs) > 1 else 0.0
        return {"mean": round(m, 4), "std": round(s, 4), "values": [round(x, 4) for x in xs]}

    return {
        "processed_tokens_per_rollout_mean": _ms(processed),
        "generated_tokens_mean": _ms(generated),
        "neg_branch_fraction": _ms(neg_frac),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--eval-iter", type=int, default=3)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--method", required=True)
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

    efficiency = _collect_efficiency(runs_root, args.num_runs, args.eval_iter)

    if values:
        m = mean(values)
        s = pstdev(values) if len(values) > 1 else 0.0
        summary = {
            "dataset": args.dataset,
            "model_tag": args.model_tag,
            "method": args.method,
            "eval_iter": iter_tag,
            "num_runs": args.num_runs,
            "n_valid": len(values),
            "mean_pass_at_1": round(m, 6),
            "std_pass_at_1": round(s, 6),
            "mean_pass_at_1_pct": round(m * 100, 4),
            "std_pass_at_1_pct": round(s * 100, 4),
            "values_pct": [round(v * 100, 4) for v in values],
            "per_run": per_run,
            "efficiency": efficiency,
        }
        md = (
            f"# {args.method} {args.num_runs}-Run Summary\n\n"
            f"- **Dataset**: {args.dataset}\n"
            f"- **Model**: {args.model_tag}\n"
            f"- **Method**: {args.method}\n"
            f"- **Metric**: {iter_tag} Pass@1\n\n"
            f"## Accuracy\n\n"
            f"**{summary['mean_pass_at_1_pct']:.2f} ± {summary['std_pass_at_1_pct']:.2f}%** "
            f"(n={summary['n_valid']})\n\n"
            f"## Efficiency ({iter_tag})\n\n"
        )
        for key, label in [
            ("processed_tokens_per_rollout_mean", "Processed tokens / rollout"),
            ("generated_tokens_mean", "Generated tokens / rollout"),
            ("neg_branch_fraction", "Neg branch / rescored fraction"),
        ]:
            entry = efficiency.get(key, {})
            mv, sv = entry.get("mean"), entry.get("std")
            if mv is not None:
                md += f"- **{label}**: {mv:.2f} ± {sv:.2f}\n"
            else:
                md += f"- **{label}**: N/A\n"
    else:
        summary = {
            "dataset": args.dataset,
            "model_tag": args.model_tag,
            "method": args.method,
            "eval_iter": iter_tag,
            "num_runs": args.num_runs,
            "n_valid": 0,
            "mean_pass_at_1_pct": None,
            "std_pass_at_1_pct": None,
            "per_run": per_run,
            "efficiency": efficiency,
        }
        md = f"# {args.method} Summary\n\nNo valid runs for {iter_tag}.\n"

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
