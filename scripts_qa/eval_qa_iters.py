#!/usr/bin/env python3
"""评估单次 RSE QA 运行的 iter0..3 EM Pass@1。"""
import argparse
import json
import re
import string
from pathlib import Path
from typing import Optional

ITER_DIRS = {0: "step1", 1: "step3", 2: "step5", 3: "step7"}


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s


def extract_qa_answer(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"(?i)final answer[:\s]+(.+?)(?:\n|$)",
        r"(?i)answer[:\s]+(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text.strip())
        if m:
            return m.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def check_em(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def eval_dir(ver_dir: Path, max_reference: int, tokenizer_path: str) -> Optional[float]:
    files = sorted(
        ver_dir.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not files:
        return None

    tokenizer = None
    if max_reference and tokenizer_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception:
            tokenizer = None

    correct = 0
    total = 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        gold = data.get("answer", "")
        if not gold:
            continue
        comps = data.get("completions", [])
        if tokenizer and max_reference:
            filtered = []
            for c in comps:
                text = c.get("text", "") if isinstance(c, dict) else str(c)
                try:
                    if len(tokenizer.encode(text)) <= 320000:
                        filtered.append(c)
                except Exception:
                    continue
            comps = filtered[:max_reference]
        if not comps:
            total += 1
            continue
        hit = False
        for c in comps:
            text = c.get("text", "") if isinstance(c, dict) else str(c)
            reasoning = c.get("reasoning_content", "") if isinstance(c, dict) else ""
            merged = f"{reasoning}\n{text}".strip()
            if check_em(extract_qa_answer(merged), gold):
                hit = True
                break
        correct += int(hit)
        total += 1

    return correct / total if total else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--out-base", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-reference", type=int, default=32)
    args = parser.parse_args()

    results = {}
    for iter_id, step in ITER_DIRS.items():
        ver_dir = Path(f"{args.out_base}_{step}/results")
        tag = f"iter{iter_id}"
        if not ver_dir.is_dir():
            results[tag] = {"pass_at_1": None, "dir": str(ver_dir)}
            continue
        p1 = eval_dir(ver_dir, args.max_reference, args.tokenizer_path)
        results[tag] = {
            "pass_at_1": p1,
            "pass_at_1_pct": None if p1 is None else round(p1 * 100, 4),
            "dir": str(ver_dir),
        }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
