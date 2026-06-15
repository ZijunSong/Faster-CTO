#!/usr/bin/env python3
"""
CB-CTO: Chunked Budgeted Contrastive Trajectory Optimization (vLLM-only).

Each reasoning step selects a chunk c from B candidate continuations:

    c* = argmax_{c in C} [ logp_pos(c|x) - g(x,C) * alpha * logp_neg(c|x) ]

g(x,C)=1 only when the chunk is high-risk under the chosen trigger; otherwise only
the positive branch is used for selection.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import SamplingParams

import cto_guided_search as cto
import vllm_efficient_cto_common as vec

logger = logging.getLogger(__name__)


def _finish_reason_from_text(tokenizer: Any, text: str, eos_token_id: Optional[int]) -> str:
    if not text:
        return "length"
    if eos_token_id is not None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids and ids[-1] == eos_token_id:
                return "stop"
        except Exception:
            pass
    return "length"


def run_cb_cto_for_question(
    llm: Any,
    tokenizer: Any,
    question_text: str,
    experience_data: Dict[str, Any],
    *,
    n_completions: int,
    max_new_tokens: int,
    alpha: float,
    temperature: float,
    top_p: float,
    top_k: int,
    chunk_size: int,
    chunk_candidates: int,
    trigger_mode: str,
    trigger_threshold: float,
    embed_model_path: str,
    max_model_len: Optional[int],
    vllm_score_batch_size: int,
    vllm_max_score_prompt_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prompt_pos, prompt_neg, pitfall_texts = vec.build_pos_neg_prompts(
        tokenizer, question_text, experience_data
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)

    completions: List[Dict[str, Any]] = []
    all_stats: List[Dict[str, Any]] = []

    for _ in range(max(1, n_completions)):
        generated = ""
        n_chunks = 0
        n_triggered = 0
        pos_forwards = 0
        neg_forwards = 0
        processed_tokens = 0
        gen_tokens = 0

        while gen_tokens < max_new_tokens:
            current_pos = prompt_pos + generated
            current_neg = prompt_neg + generated
            remain = max_new_tokens - gen_tokens
            this_chunk = min(int(chunk_size), int(remain))
            if this_chunk <= 0:
                break

            gen_params = SamplingParams(
                n=max(1, int(chunk_candidates)),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=this_chunk,
                logprobs=5,
            )
            gen_out = llm.generate([current_pos], gen_params)[0].outputs
            pos_forwards += 1
            processed_tokens += cto._token_count(tokenizer, current_pos)

            cand_texts: List[str] = []
            cand_pos_scores: List[float] = []
            cand_logprobs: List[Any] = []
            for out in gen_out:
                text = out.text or ""
                cand_texts.append(text)
                cand_logprobs.append(getattr(out, "logprobs", None))
                cand_pos_scores.append(vec.sum_output_logprobs(getattr(out, "logprobs", None)))

            if not cand_texts:
                break

            prefix_for_risk = generated
            high_risk = any(
                vec.chunk_is_high_risk(
                    trigger_mode,
                    chunk_text=text,
                    prefix_text=prefix_for_risk,
                    logprobs=lp,
                    all_candidate_texts=cand_texts,
                    pitfall_texts=pitfall_texts,
                    embed_model_path=embed_model_path,
                    threshold=trigger_threshold,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    tokenizer=tokenizer,
                )
                for text, lp in zip(cand_texts, cand_logprobs)
            )
            if high_risk:
                n_triggered += 1
                neg_scores = vec.score_suffixes_batch(
                    llm,
                    tokenizer,
                    current_neg,
                    cand_texts,
                    max_model_len=max_model_len,
                    vllm_score_batch_size=vllm_score_batch_size,
                    vllm_max_score_prompt_tokens=vllm_max_score_prompt_tokens,
                )
                neg_forwards += 1
                processed_tokens += sum(
                    cto._token_count(tokenizer, current_neg + t) + 1 for t in cand_texts
                )
                contrastive = [
                    pos - alpha * neg for pos, neg in zip(cand_pos_scores, neg_scores)
                ]
                best_idx = int(max(range(len(contrastive)), key=lambda i: contrastive[i]))
            else:
                best_idx = int(max(range(len(cand_pos_scores)), key=lambda i: cand_pos_scores[i]))

            chosen = cand_texts[best_idx]
            chosen_ids = getattr(gen_out[best_idx], "token_ids", []) or []
            processed_tokens += cto._token_count(tokenizer, current_pos)
            processed_tokens += len(chosen_ids)
            gen_tokens += len(chosen_ids)
            generated += chosen
            n_chunks += 1

            if _finish_reason_from_text(tokenizer, chosen, eos_id) == "stop":
                break
            if not chosen.strip():
                break

        text_raw = generated
        final_text, reasoning = cto.extract_thinking(text_raw)
        rollout_stats = {
            "backend": "vllm_cb_cto",
            "trigger_mode": trigger_mode,
            "trigger_threshold": float(trigger_threshold),
            "chunk_size": int(chunk_size),
            "chunk_candidates": int(chunk_candidates),
            "n_chunks": n_chunks,
            "neg_chunk_fraction": float(n_triggered / max(n_chunks, 1)),
            "pos_forwards": pos_forwards,
            "neg_forwards": neg_forwards,
            "generated_tokens": cto._token_count(tokenizer, text_raw),
            "processed_tokens_total": int(processed_tokens),
            "processed_tokens_per_rollout_mean": int(processed_tokens),
        }
        all_stats.append(rollout_stats)
        completions.append(
            {
                "text": final_text,
                "reasoning_content": reasoning,
                "tokens": rollout_stats["generated_tokens"],
                "finish_reason": _finish_reason_from_text(tokenizer, text_raw, eos_id),
                "cb_cto_stats": rollout_stats,
            }
        )

    n_comp = max(len(all_stats), 1)
    compute = {
        "backend": "vllm_cb_cto",
        "trigger_mode": trigger_mode,
        "trigger_threshold": float(trigger_threshold),
        "chunk_size": int(chunk_size),
        "chunk_candidates": int(chunk_candidates),
        "neg_branch_fraction": float(
            sum(s["neg_chunk_fraction"] for s in all_stats) / n_comp
        ),
        "generated_tokens_mean": float(
            sum(s["generated_tokens"] for s in all_stats) / n_comp
        ),
        "processed_tokens_total": int(sum(s["processed_tokens_total"] for s in all_stats)),
        "processed_tokens_per_rollout_mean": int(
            sum(s["processed_tokens_total"] for s in all_stats) / n_comp
        ),
    }
    return completions, compute


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CB-CTO: Chunked Budgeted Contrastive Trajectory Optimization (vLLM)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--experience-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-completions", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--chunk-candidates", type=int, default=4)
    parser.add_argument(
        "--cb-cto-trigger",
        type=str,
        required=True,
        choices=list(vec.TRIGGER_MODES),
    )
    parser.add_argument("--cb-cto-threshold", type=float, default=None)
    vec.add_experience_args(parser)
    vec.add_vllm_args(parser)
    args = parser.parse_args()

    trigger_threshold = (
        args.cb_cto_threshold
        if args.cb_cto_threshold is not None
        else vec.DEFAULT_THRESHOLDS[args.cb_cto_trigger]
    )

    task_type = vec.setup_task_prefixes(args)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    experience_path = Path(args.experience_dir)

    questions = cto.load_jsonl(args.input)
    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions = questions[args.start_idx:end_idx]
    logger.info(
        "CB-CTO trigger=%s threshold=%.4f chunk=%d B=%d | %s | %d questions",
        args.cb_cto_trigger,
        trigger_threshold,
        args.chunk_size,
        args.chunk_candidates,
        task_type,
        len(questions),
    )

    cross_query_parsed, cross_query_doc_emb = vec.init_cross_query_pool(args, experience_path)
    pending = vec.prepare_pending_questions(args, questions, output_path)
    if not pending:
        logger.info("All questions already completed.")
        return

    llm, tokenizer, max_model_len = vec.init_vllm(args)

    for item in tqdm(pending, desc=f"CB-CTO({args.cb_cto_trigger})", unit="q"):
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

        completions, compute = run_cb_cto_for_question(
            llm,
            tokenizer,
            question_text,
            experience_data,
            n_completions=args.n_completions,
            max_new_tokens=args.max_tokens,
            alpha=args.alpha,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            chunk_candidates=args.chunk_candidates,
            trigger_mode=args.cb_cto_trigger,
            trigger_threshold=trigger_threshold,
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
            extra_fields={"cb_cto_compute": compute, "cto_compute": compute},
        )

    logger.info("Done. Results in %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
