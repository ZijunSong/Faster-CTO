#!/usr/bin/env python3
"""将 run*_eval_summary.json 整理为宽表 CSV（双层表头，如图示格式）。"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

ITERS = ["iter0", "iter1", "iter2", "iter3"]
RUNS = [1, 2, 3]


def load_runs(runs_root: Path, num_runs: int = 3) -> dict[int, dict]:
    runs = {}
    for run_id in range(1, num_runs + 1):
        path = runs_root / f"run{run_id}_eval_summary.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            runs[run_id] = json.load(f)
    return runs


def iter_stats(runs: dict[int, dict], iter_key: str) -> tuple[float, float]:
    pcts = [runs[r][iter_key]["pass_at_1_pct"] for r in sorted(runs)]
    vals = [p / 100.0 for p in pcts]
    m = mean(vals)
    s = pstdev(vals) if len(vals) > 1 else 0.0
    return m * 100, s * 100


def fmt_mean_std(m: float, s: float) -> str:
    return f"{m:.1f}±{s:.1f}"


def fmt_pct(p: float) -> str:
    return f"{p:.1f}"


def build_row(
    model_tag: str,
    dataset: str,
    runs: dict[int, dict],
    gpu_group: str = "",
) -> list[str]:
    row = [model_tag]
    if gpu_group:
        row.append(gpu_group)

    for iter_key in ITERS:
        m, s = iter_stats(runs, iter_key)
        row.append(fmt_mean_std(m, s))

    for run_id in RUNS:
        for iter_key in ITERS:
            row.append(fmt_pct(runs[run_id][iter_key]["pass_at_1_pct"]))

    return row


def header_rows(dataset: str, include_gpu: bool = False) -> tuple[list[str], list[str]]:
    groups = [f"{dataset}(mean±std)"] + [f"{dataset}(run{r})" for r in RUNS]
    row1 = ["model_tag"]
    if include_gpu:
        row1.append("gpu_group")
    for g in groups:
        row1.extend([g, "", "", ""])

    row2 = [""]
    if include_gpu:
        row2.append("")
    for _ in groups:
        row2.extend(ITERS)

    return row1, row2


def write_wide_csv(
    csv_path: Path,
    dataset: str,
    rows: list[list[str]],
    include_gpu: bool = False,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    h1, h2 = header_rows(dataset, include_gpu=include_gpu)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(h1)
        w.writerow(h2)
        for row in rows:
            w.writerow(row)


def collect_dataset_models(runs_qa_root: Path, dataset: str) -> list[Path]:
    dataset_dir = runs_qa_root / dataset
    if not dataset_dir.exists():
        return []
    out = []
    for model_dir in sorted(dataset_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if all((model_dir / f"run{r}_eval_summary.json").exists() for r in RUNS):
            out.append(model_dir)
    return out


def model_tag_from_dir(model_dir: Path) -> str:
    name = model_dir.name
    for suffix in ("_GPU0", "_GPU1", "_GPU2", "_GPU3", "_GPU4", "_GPU5", "_GPU6", "_GPU7",
                   "_GPU0-1", "_GPU0-3", "_GPU2-3", "_GPU4-7"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def gpu_group_from_dir(model_dir: Path) -> str:
    name = model_dir.name
    for prefix in ("_GPU",):
        idx = name.rfind(prefix)
        if idx != -1:
            return name[idx + 1 :]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset", default="BambooQA")
    parser.add_argument("--runs-root", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    runs_qa_root = project_root / "runs_qa"
    results_qa_root = project_root / "results_qa"

    if args.runs_root:
        model_dirs = [Path(args.runs_root).resolve()]
        dataset = args.dataset
    else:
        dataset = args.dataset
        model_dirs = collect_dataset_models(runs_qa_root, dataset)

    if not model_dirs:
        raise SystemExit(f"No complete 3-run results found for dataset={dataset}")

    rows = []
    for model_dir in model_dirs:
        runs = load_runs(model_dir)
        if len(runs) != 3:
            continue
        tag = model_tag_from_dir(model_dir)
        gpu = gpu_group_from_dir(model_dir)
        rows.append(build_row(tag, dataset, runs, gpu_group=gpu))

    dataset_csv = results_qa_root / f"{dataset}_results.csv"
    write_wide_csv(dataset_csv, dataset, rows, include_gpu=True)
    print(f"Wrote dataset CSV: {dataset_csv}")

    for model_dir in model_dirs:
        runs = load_runs(model_dir)
        if len(runs) != 3:
            continue
        tag = model_tag_from_dir(model_dir)
        gpu = gpu_group_from_dir(model_dir)
        row = build_row(tag, dataset, runs)
        out = model_dir / f"{model_dir.name}_results_wide.csv"
        if args.output and len(model_dirs) == 1:
            out = Path(args.output)
        write_wide_csv(out, dataset, [row], include_gpu=False)
        print(f"Wrote model CSV: {out}")


if __name__ == "__main__":
    main()
