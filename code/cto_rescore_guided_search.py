#!/usr/bin/env python3
"""
CTO-Rescore: Sparse Negative Reranking (vLLM-only).

Pipeline per question:
  1. Positive-only generation of N trajectories
  2. Select M candidates (top positive score or high-risk)
  3. Teacher-forced negative scoring on M suffixes
  4. Rank by S(y) = logp_pos(y) - alpha * logp_neg(y)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from vllm import SamplingParams

import cto_guided_search as cto
import vllm_efficient_cto_common as vec

logger = logging.getLogger(__name__)


def _select_rescore_indices(
    *,
    selection: str,
    pos_scores: List[float],
    candidate_texts: List[str],
    candidate_logprobs: List[Any],
    tokenizer: Any,
    pitfall_texts: List[str],
    embed_model_path: str,
    rescore_m: int,
    risk_threshold: float,
) -> List[int]:
    n = len(candidate_texts)
    m = min(max(1, int(rescore_m)), n)
    if selection == "top_pos":
        order = sorted(range(n), key=lambda i: pos_scores[i], reverse=True)
        return order[:m]
    if selection == "random":
        import random

        return random.sample(list(range(n)), m)
    if selection == "high_risk":
        risky: List[Tuple[float, int]] = []
        for i, text in enumerate(candidate_texts):
            risk = 0.0
            if candidate_logprobs[i]:
                ents = [
                    vec._normalized_entropy_from_step(s)
                    for s in candidate_logprobs[i]
                    if s
                ]
                if ents:
                    risk = max(risk, float(np.mean(ents)))
            risk = max(
                risk,
                vec._max_pitfall_similarity(text, pitfall_texts, embed_model_path),
            )
            if risk >= risk_threshold:
                risky.append((risk, i))
        risky.sort(key=lambda x: (-x[0], pos_scores[x[1]]))
        risky_idx = [i for _, i in risky]
        if len(risky_idx) >= m:
            return risky_idx[:m]
        order = sorted(range(n), key=lambda i: pos_scores[i])
        chosen = list(dict.fromkeys(risky_idx + order))
        return chosen[:m]
    raise ValueError(f"Unknown rescore selection: {selection}")


def run_cto_rescore_for_question(
    llm: Any,
    tokenizer: Any,
    question_text: str,
    experience_data: Dict[str, Any],
    *,
    n_completions: int,
    n_generate: int,
    rescore_m: int,
    rescore_selection: str,
    risk_threshold: float,
    max_new_tokens: int,
    alpha: float,
    temperature: float,
    top_p: float,
    top_k: int,
    embed_model_path: str,
    max_model_len: Optional[int],
    vllm_score_batch_size: int,
    vllm_max_score_prompt_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prompt_pos, prompt_neg, pitfall_texts = vec.build_pos_neg_prompts(
        tokenizer, question_text, experience_data
    )

    n_gen = max(int(n_generate), int(n_completions), 1)
    gen_params = SamplingParams(
        n=n_gen,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        logprobs=1,
    )
    gen_out = llm.generate([prompt_pos], gen_params)[0].outputs
    candidates = [o.text or "" for o in gen_out]
    pos_scores = [vec.sum_output_logprobs(getattr(o, "logprobs", None)) for o in gen_out]
    cand_logprobs = [getattr(o, "logprobs", None) for o in gen_out]
    gen_tokens_total = int(sum(len(getattr(o, "token_ids", []) or []) for o in gen_out))

    selected_idx = _select_rescore_indices(
        selection=rescore_selection,
        pos_scores=pos_scores,
        candidate_texts=candidates,
        candidate_logprobs=cand_logprobs,
        tokenizer=tokenizer,
        pitfall_texts=pitfall_texts,
        embed_model_path=embed_model_path,
        rescore_m=rescore_m,
        risk_threshold=risk_threshold,
    )
    selected_texts = [candidates[i] for i in selected_idx]
    neg_scores_map: Dict[int, float] = {}
    if selected_texts:
        neg_scores = vec.score_suffixes_batch(
            llm,
            tokenizer,
            prompt_neg,
            selected_texts,
            max_model_len=max_model_len,
            vllm_score_batch_size=vllm_score_batch_size,
            vllm_max_score_prompt_tokens=vllm_max_score_prompt_tokens,
        )
        neg_scores_map = {selected_idx[i]: neg_scores[i] for i in range(len(selected_idx))}

    scored: List[Tuple[float, int]] = []
    for i, text in enumerate(candidates):
        neg = neg_scores_map.get(i, 0.0)
        final = float(pos_scores[i] - alpha * neg)
        scored.append((final, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[: max(1, int(n_completions))]

    prompt_pos_len = cto._token_count(tokenizer, prompt_pos)
    prompt_neg_len = cto._token_count(tokenizer, prompt_neg)
    scoring_pos_tokens = 0
    scoring_neg_tokens = 0
    for i in selected_idx:
        scoring_neg_tokens += cto._token_count(tokenizer, prompt_neg + candidates[i]) + 1
    for _, idx in picked:
        scoring_pos_tokens += cto._token_count(tokenizer, prompt_pos + candidates[idx]) + 1

    processed_tokens_total = (
        prompt_pos_len
        + gen_tokens_total
        + scoring_pos_tokens
        + scoring_neg_tokens
    )
    compute = {
        "backend": "vllm_cto_rescore",
        "n_generate": int(n_gen),
        "rescore_m": int(len(selected_idx)),
        "rescore_selection": rescore_selection,
        "neg_branch_fraction": float(len(selected_idx) / max(n_gen, 1)),
        "prompt_pos_tokens": int(prompt_pos_len),
        "prompt_neg_tokens": int(prompt_neg_len),
        "candidate_generation_new_tokens": int(gen_tokens_total),
        "processed_tokens_total": int(processed_tokens_total),
        "processed_tokens_per_rollout_mean": int(processed_tokens_total / max(n_completions, 1)),
        "generated_tokens_mean": float(gen_tokens_total / max(n_gen, 1)),
    }

    completions: List[Dict[str, Any]] = []
    for score, idx in picked:
        text_raw = candidates[idx]
        final_text, reasoning = cto.extract_thinking(text_raw)
        completions.append(
            {
                "text": final_text,
                "reasoning_content": reasoning,
                "tokens": cto._token_count(tokenizer, text_raw),
                "finish_reason": "score_rerank",
                "cto_score": score,
                "pos_score": pos_scores[idx],
                "neg_score": neg_scores_map.get(idx, 0.0),
                "rescored": idx in neg_scores_map,
            }
        )
    return completions, compute


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CTO-Rescore: sparse negative reranking (vLLM)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--experience-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-completions", type=int, default=1)
    parser.add_argument("--n-generate", type=int, default=32)
    parser.add_argument("--rescore-m", type=int, default=8)
    parser.add_argument(
        "--rescore-selection",
        type=str,
        default="top_pos",
        choices=["top_pos", "high_risk", "random"],
    )
    parser.add_argument(
        "--rescore-risk-threshold",
        type=float,
        default=0.85,
        help="Used when --rescore-selection=high_risk",
    )
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    vec.add_experience_args(parser)
    vec.add_vllm_args(parser)
    args = parser.parse_args()

    task_type = vec.setup_task_prefixes(args)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    experience_path = Path(args.experience_dir)

    questions = cto.load_jsonl(args.input)
    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions = questions[args.start_idx:end_idx]
    logger.info(
        "CTO-Rescore N=%d M=%d selection=%s | %s | %d questions",
        args.n_generate,
        args.rescore_m,
        args.rescore_selection,
        task_type,
        len(questions),
    )

    cross_query_parsed, cross_query_doc_emb = vec.init_cross_query_pool(args, experience_path)
    pending = vec.prepare_pending_questions(args, questions, output_path)
    if not pending:
        logger.info("All questions already completed.")
        return

    llm, tokenizer, max_model_len = vec.init_vllm(args)

    for item in tqdm(pending, desc="CTO-Rescore", unit="q"):
        orig_idx = item["index"]
        question_text = item.get("question", "")
        experience_data = vec.load_experience_for_question(
            args,
            experience_path,
            orig_idx,
            question_text,
            cross_query_parsed,
            cross_query_doc_emb,
        )
        if not experience_data:
            logger.warning("No experience for question %s, skipping.", orig_idx)
            continue

        completions, compute = run_cto_rescore_for_question(
            llm,
            tokenizer,
            question_text,
            experience_data,
            n_completions=args.n_completions,
            n_generate=args.n_generate,
            rescore_m=args.rescore_m,
            rescore_selection=args.rescore_selection,
            risk_threshold=args.rescore_risk_threshold,
            max_new_tokens=args.max_tokens,
            alpha=args.alpha,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            embed_model_path=args.retrieval_embedding_model,
            max_model_len=max_model_len,
            vllm_score_batch_size=args.vllm_score_batch_size,
            vllm_max_score_prompt_tokens=args.vllm_max_score_prompt_tokens,
        )
        cto.save_result(
            output_path,
            orig_idx,
            item,
            completions,
            extra_fields={"cto_rescore_compute": compute, "cto_compute": compute},
        )

    logger.info("Done. Results in %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
