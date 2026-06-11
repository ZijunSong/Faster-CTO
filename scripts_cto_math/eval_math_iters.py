#!/usr/bin/env python3
"""评估单次 CTO/RSE math 运行的 iter0..3 Pass@1。"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ITER_DIRS = {0: "step1", 1: "step3", 2: "step5", 3: "step7"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--out-base", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-reference", type=int, default=32)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    eval_script = root / "eval" / "calculate_pass_at_k_from_completions.py"
    results = {}

    for iter_id, step in ITER_DIRS.items():
        ver_dir = Path(f"{args.out_base}_{step}/results")
        tag = f"iter{iter_id}"
        if not ver_dir.is_dir():
            results[tag] = {"pass_at_1": None, "dir": str(ver_dir)}
            continue

        out_json = ver_dir / "pass_at_k.json"
        subprocess.run(
            [
                sys.executable,
                str(eval_script),
                "--verification_dir",
                str(ver_dir),
                "--k_values",
                "1",
                "--output_file",
                str(out_json),
                "--max_reference",
                str(args.max_reference),
                "--tokenizer_path",
                args.tokenizer_path,
            ],
            check=True,
        )
        with open(out_json, encoding="utf-8") as f:
            metrics = json.load(f)
        p1 = metrics.get("pass_at_k", {}).get("pass@1")
        results[tag] = {
            "pass_at_1": p1,
            "pass_at_1_pct": None if p1 is None else round(p1 * 100, 4),
            "dir": str(ver_dir),
        }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
