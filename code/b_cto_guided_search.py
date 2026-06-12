#!/usr/bin/env python3
"""
B-CTO (Budgeted Contrastive Trajectory Optimization).

Token-level gating: only run the negative branch when a risk trigger fires.
s_i = l_pos,i - g_j * alpha * l_neg,i,  g_j in {0, 1}.

Triggers (four research signals):
  - entropy: normalized softmax entropy of logits_pos > threshold
  - margin: top-1 minus top-2 logit margin < threshold
  - failure_pattern: prefix embedding similarity to critical_pitfalls > threshold
  - rollout_disagreement: argmax(logits_pos) != stochastic sample(logits_pos)
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import cto_guided_search as cto
from task_prompts import add_task_args, get_cto_prefixes, resolve_task_type

logger = logging.getLogger(__name__)

TRIGGER_MODES = ("entropy", "margin", "failure_pattern", "rollout_disagreement")


def _normalized_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    probs = probs.clamp(min=1e-12)
    ent = -(probs * probs.log()).sum().item()
    vocab = max(int(logits.numel()), 2)
    return float(ent / math.log(vocab))


def _top12_margin(logits: torch.Tensor) -> float:
    top2 = torch.topk(logits.float(), k=min(2, logits.numel()), dim=-1).values
    if top2.numel() < 2:
        return float("inf")
    return float(top2[0] - top2[1])


def _argmax_token(logits: torch.Tensor) -> int:
    return int(logits.argmax(dim=-1).item())


def _stochastic_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    pad_token_id: Optional[int],
) -> int:
    return cto.sample_from_logits(
        logits.unsqueeze(0),
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        top_k=top_k,
        pad_token_id=pad_token_id,
    )


def _prefix_text(tokenizer: Any, prompt_ids: torch.LongTensor, generated: List[int]) -> str:
    if generated:
        ids = torch.cat([prompt_ids, torch.tensor(generated, dtype=prompt_ids.dtype)])
    else:
        ids = prompt_ids
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


def _max_pitfall_similarity(
    prefix_text: str,
    pitfall_texts: List[str],
    embed_model_path: str,
) -> float:
    if not pitfall_texts:
        return 0.0
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers missing; failure_pattern trigger disabled")
        return 0.0

    st = cto._EMB_ST.get(embed_model_path)
    if st is None:
        st = SentenceTransformer(embed_model_path)
        cto._EMB_ST[embed_model_path] = st

    q = prefix_text.strip()[:4000]
    if not q:
        return 0.0
    qv = st.encode([q], normalize_embeddings=True)
    pv = st.encode(pitfall_texts[:32], normalize_embeddings=True)
    sim = np.dot(pv, qv.T).flatten()
    return float(sim.max()) if sim.size else 0.0


def compute_trigger(
    mode: str,
    logits_pos: torch.Tensor,
    *,
    threshold: float,
    temperature: float,
    top_p: float,
    top_k: int,
    pad_token_id: Optional[int],
    prefix_text: str = "",
    pitfall_texts: Optional[List[str]] = None,
    embed_model_path: str = "",
) -> bool:
    if mode == "entropy":
        return _normalized_entropy(logits_pos) > threshold
    if mode == "margin":
        return _top12_margin(logits_pos) < threshold
    if mode == "failure_pattern":
        sim = _max_pitfall_similarity(
            prefix_text,
            pitfall_texts or [],
            embed_model_path,
        )
        return sim > threshold
    if mode == "rollout_disagreement":
        greedy = _argmax_token(logits_pos)
        sampled = _stochastic_token(
            logits_pos,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=pad_token_id,
        )
        return greedy != sampled
    raise ValueError(f"Unknown B-CTO trigger mode: {mode}")


def _neg_forward_step(
    model: torch.nn.Module,
    token_batch: torch.LongTensor,
    past_key_values_neg: Any,
) -> Tuple[torch.Tensor, Any]:
    with torch.no_grad():
        if past_key_values_neg is None:
            out_neg = model(input_ids=token_batch, use_cache=True, return_dict=True)
        else:
            out_neg = model(
                input_ids=token_batch,
                past_key_values=past_key_values_neg,
                use_cache=True,
                return_dict=True,
            )
    logits_neg = out_neg.logits[:, -1, :].squeeze(0)
    return logits_neg, out_neg.past_key_values


def b_cto_decode_single(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids_pos: torch.LongTensor,
    input_ids_neg: torch.LongTensor,
    max_new_tokens: int,
    alpha: float,
    plausibility_top_k: int,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    device: torch.device,
    trigger_mode: str,
    trigger_threshold: float,
    pitfall_texts: Optional[List[str]] = None,
    embed_model_path: str = "",
) -> Tuple[List[int], Dict[str, Any]]:
    """Budgeted CTO decode with lazy negative-branch forwards."""
    model.eval()
    generated: List[int] = []
    current_pos = input_ids_pos.to(device).unsqueeze(0)
    current_neg = input_ids_neg.to(device).unsqueeze(0)
    past_key_values_pos = None
    past_key_values_neg = None
    deferred_neg_tokens: List[torch.LongTensor] = []

    prompt_pos_len = int(input_ids_pos.numel())
    prompt_neg_len = int(input_ids_neg.numel())
    decode_steps = 0
    neg_trigger_steps = 0
    processed_tokens = 0
    neg_forwards = 0
    pos_forwards = 0

    for _ in range(max_new_tokens):
        if past_key_values_pos is None:
            with torch.no_grad():
                out_pos = model(
                    input_ids=current_pos,
                    use_cache=True,
                    return_dict=True,
                )
        else:
            with torch.no_grad():
                out_pos = model(
                    input_ids=current_pos,
                    past_key_values=past_key_values_pos,
                    use_cache=True,
                    return_dict=True,
                )
        logits_pos = out_pos.logits[:, -1, :].squeeze(0)
        past_key_values_pos = out_pos.past_key_values
        pos_forwards += 1
        processed_tokens += int(current_pos.numel())

        prefix_text = _prefix_text(tokenizer, input_ids_pos, generated)
        g_j = compute_trigger(
            trigger_mode,
            logits_pos,
            threshold=trigger_threshold,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=pad_token_id,
            prefix_text=prefix_text,
            pitfall_texts=pitfall_texts,
            embed_model_path=embed_model_path,
        )

        if g_j:
            neg_trigger_steps += 1
            if past_key_values_neg is None and not deferred_neg_tokens:
                with torch.no_grad():
                    out_neg = model(
                        input_ids=current_neg,
                        use_cache=True,
                        return_dict=True,
                    )
                logits_neg = out_neg.logits[:, -1, :].squeeze(0)
                past_key_values_neg = out_neg.past_key_values
                neg_forwards += 1
                processed_tokens += int(current_neg.numel())
            else:
                if past_key_values_neg is None:
                    with torch.no_grad():
                        out_neg = model(
                            input_ids=input_ids_neg.to(device).unsqueeze(0),
                            use_cache=True,
                            return_dict=True,
                        )
                    past_key_values_neg = out_neg.past_key_values
                    neg_forwards += 1
                    processed_tokens += prompt_neg_len
                for dt in deferred_neg_tokens:
                    _, past_key_values_neg = _neg_forward_step(
                        model, dt, past_key_values_neg
                    )
                    neg_forwards += 1
                    processed_tokens += 1
                deferred_neg_tokens = []
                logits_neg, past_key_values_neg = _neg_forward_step(
                    model, current_pos, past_key_values_neg
                )
                neg_forwards += 1
                processed_tokens += int(current_pos.numel())

            logits_cto = logits_pos - alpha * logits_neg
            logits_cto = cto.apply_plausibility_constraint(
                logits_pos, logits_cto, plausibility_top_k
            )
        else:
            logits_cto = logits_pos

        next_token = cto.sample_from_logits(
            logits_cto.unsqueeze(0),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=pad_token_id,
        )
        generated.append(next_token)
        decode_steps += 1

        if eos_token_id is not None and next_token == eos_token_id:
            break

        next_t = torch.tensor([[next_token]], dtype=current_pos.dtype, device=device)
        current_pos = next_t
        if g_j:
            current_neg = next_t
        else:
            deferred_neg_tokens.append(next_t)

    stats = {
        "backend": "hf_b_cto",
        "trigger_mode": trigger_mode,
        "trigger_threshold": float(trigger_threshold),
        "prompt_pos_tokens": prompt_pos_len,
        "prompt_neg_tokens": prompt_neg_len,
        "decode_steps": decode_steps,
        "generated_tokens": len(generated),
        "neg_trigger_steps": neg_trigger_steps,
        "neg_branch_fraction": float(neg_trigger_steps / max(decode_steps, 1)),
        "pos_forwards": pos_forwards,
        "neg_forwards": neg_forwards,
        "processed_tokens_total": int(processed_tokens),
        "processed_tokens_per_rollout_mean": int(processed_tokens),
    }
    return generated, stats


def run_b_cto_for_question(
    model: Any,
    tokenizer: Any,
    question_text: str,
    experience_data: Dict[str, Any],
    n_completions: int,
    max_new_tokens: int,
    alpha: float,
    plausibility_top_k: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: torch.device,
    trigger_mode: str,
    trigger_threshold: float,
    embed_model_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    props_str = cto.build_p_pos_content(experience_data)
    pitfalls_str = cto.build_p_neg_content(experience_data)
    pitfall_texts = [
        p for p in (experience_data.get("critical_pitfalls") or []) if isinstance(p, str)
    ]

    system_pos = (cto.P_POS_SYSTEM_PREFIX + props_str).strip() if props_str else ""
    system_neg = (
        (cto.P_NEG_SYSTEM_PREFIX + pitfalls_str).strip()
        if pitfalls_str
        else "Do not repeat the following errors:\n"
    )

    messages_pos = [
        {"role": "system", "content": system_pos or cto._CTO_FALLBACK_SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]
    messages_neg = [
        {"role": "system", "content": system_neg or "Please reason step by step."},
        {"role": "user", "content": question_text},
    ]

    input_ids_pos = tokenizer.apply_chat_template(
        messages_pos,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).squeeze(0)
    input_ids_neg = tokenizer.apply_chat_template(
        messages_neg,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).squeeze(0)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id

    completions: List[Dict[str, Any]] = []
    agg_processed = 0
    agg_generated = 0
    agg_neg_frac = 0.0

    for _ in range(n_completions):
        new_ids, stats = b_cto_decode_single(
            model=model,
            tokenizer=tokenizer,
            input_ids_pos=input_ids_pos,
            input_ids_neg=input_ids_neg,
            max_new_tokens=max_new_tokens,
            alpha=alpha,
            plausibility_top_k=plausibility_top_k,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            device=device,
            trigger_mode=trigger_mode,
            trigger_threshold=trigger_threshold,
            pitfall_texts=pitfall_texts,
            embed_model_path=embed_model_path,
        )
        text_raw = tokenizer.decode(new_ids, skip_special_tokens=False) if new_ids else ""
        final_text, reasoning = cto.extract_thinking(text_raw)
        completions.append(
            {
                "text": final_text,
                "reasoning_content": reasoning,
                "tokens": len(new_ids),
                "finish_reason": "stop"
                if (eos_id and new_ids and new_ids[-1] == eos_id)
                else "length",
                "b_cto_stats": stats,
            }
        )
        agg_processed += stats["processed_tokens_total"]
        agg_generated += stats["generated_tokens"]
        agg_neg_frac += stats["neg_branch_fraction"]

    n_comp = max(n_completions, 1)
    b_cto_compute = {
        "backend": "hf_b_cto",
        "trigger_mode": trigger_mode,
        "trigger_threshold": float(trigger_threshold),
        "prompt_pos_tokens": int(input_ids_pos.numel()),
        "prompt_neg_tokens": int(input_ids_neg.numel()),
        "neg_branch_fraction": float(agg_neg_frac / n_comp),
        "generated_tokens_mean": float(agg_generated / n_comp),
        "processed_tokens_total": int(agg_processed),
        "processed_tokens_per_rollout_mean": int(agg_processed / n_comp),
    }
    return completions, b_cto_compute


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B-CTO: Budgeted Contrastive Trajectory Optimization (HF token-level gating)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--experience-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-experience-completions", type=int, default=5)
    parser.add_argument("--n-completions", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--plausibility-top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--b-cto-trigger",
        type=str,
        required=True,
        choices=list(TRIGGER_MODES),
        help="Risk trigger for negative branch gating",
    )
    parser.add_argument(
        "--b-cto-threshold",
        type=float,
        default=None,
        help="Trigger threshold (mode-specific). Defaults: entropy=0.85, margin=1.0, failure_pattern=0.55",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])
    parser.add_argument(
        "--experience-retrieval",
        type=str,
        default="first",
        choices=[
            "first",
            "embedding",
            "embedding_rerank",
            "cross_query_embedding",
            "cross_query_embedding_rerank",
        ],
    )
    parser.add_argument(
        "--retrieval-embedding-model",
        type=str,
        default="/data/ppnm/models/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--retrieval-rerank-model",
        type=str,
        default=cto._DEFAULT_RETRIEVAL_RERANK_MODEL,
    )
    parser.add_argument("--retrieval-rerank-pool-mult", type=int, default=4)
    parser.add_argument("--max-aggregated-propositions", type=int, default=0)
    parser.add_argument("--max-aggregated-pitfalls", type=int, default=0)
    parser.add_argument(
        "--experience-aggregation",
        type=str,
        default="legacy_sorted",
        choices=["legacy_sorted", "flat_top_k", "semantic_merge", "recency_aware"],
    )
    parser.add_argument("--aggregation-semantic-threshold", type=float, default=0.82)
    parser.add_argument("--aggregation-recency-head-fraction", type=float, default=0.55)
    add_task_args(parser)
    args = parser.parse_args()

    default_thresholds = {
        "entropy": 0.85,
        "margin": 1.0,
        "failure_pattern": 0.55,
        "rollout_disagreement": 0.5,
    }
    trigger_threshold = (
        args.b_cto_threshold
        if args.b_cto_threshold is not None
        else default_thresholds[args.b_cto_trigger]
    )

    task_type = resolve_task_type(
        dataset=args.dataset,
        task_type=args.task_type,
        input_path=args.input,
    )
    cto.P_POS_SYSTEM_PREFIX, cto.P_NEG_SYSTEM_PREFIX, cto._CTO_FALLBACK_SYSTEM_PROMPT = get_cto_prefixes(
        task_type
    )

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    experience_path = Path(args.experience_dir)

    questions = cto.load_jsonl(args.input)
    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions = questions[args.start_idx:end_idx]
    logger.info(
        "B-CTO trigger=%s threshold=%.4f | %d questions [%d, %d)",
        args.b_cto_trigger,
        trigger_threshold,
        len(questions),
        args.start_idx,
        end_idx,
    )

    cross_query_parsed = None
    cross_query_doc_emb = None
    if args.experience_retrieval.startswith("cross_query"):
        idx2q = cto.load_index_to_question_map(args.input)
        gp = cto.build_global_experience_pool(experience_path, idx2q)
        cross_query_parsed = cto.build_cross_query_parsed(gp)
        cross_query_doc_emb = cto.precompute_doc_embeddings_for_parsed(
            cross_query_parsed,
            args.retrieval_embedding_model,
        )

    pending = []
    for i, item in enumerate(questions):
        orig_idx = args.start_idx + i
        item["index"] = orig_idx
        out_file = output_path / f"{orig_idx}.json"
        if args.force_rerun:
            pending.append(item)
            continue
        existing = cto.load_existing_output(out_file)
        if len(existing) >= args.n_completions:
            continue
        pending.append(item)

    if not pending:
        logger.info("All questions already completed.")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
    dtype = dtype_map.get(args.dtype, "auto")
    load_kw: Dict[str, Any] = {
        "torch_dtype": dtype if isinstance(dtype, str) else dtype,
        "trust_remote_code": True,
    }
    if args.device_map:
        load_kw["device_map"] = args.device_map
    else:
        load_kw["device_map"] = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    logger.info("Loading HF model for B-CTO: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    device = next(model.parameters()).device

    for item in tqdm(pending, total=len(pending), desc=f"B-CTO({args.b_cto_trigger})", unit="q"):
        orig_idx = item["index"]
        question_text = item.get("question", "")
        experience_data = cto.load_and_aggregate_raw_experiences(
            experience_path,
            orig_idx,
            args.n_experience_completions,
            question_text=question_text,
            retrieval=args.experience_retrieval,
            embed_model_path=args.retrieval_embedding_model,
            rerank_model_name=args.retrieval_rerank_model,
            rerank_pool_mult=args.retrieval_rerank_pool_mult,
            cross_query_parsed=cross_query_parsed,
            cross_query_doc_emb=cross_query_doc_emb,
            max_aggregated_propositions=args.max_aggregated_propositions,
            max_aggregated_pitfalls=args.max_aggregated_pitfalls,
            experience_aggregation=args.experience_aggregation,
            aggregation_semantic_threshold=args.aggregation_semantic_threshold,
            aggregation_recency_head_fraction=args.aggregation_recency_head_fraction,
        )
        if not experience_data:
            logger.warning("No experience for question %s, skipping.", orig_idx)
            continue

        completions, b_cto_compute = run_b_cto_for_question(
            model=model,
            tokenizer=tokenizer,
            question_text=question_text,
            experience_data=experience_data,
            n_completions=args.n_completions,
            max_new_tokens=args.max_tokens,
            alpha=args.alpha,
            plausibility_top_k=args.plausibility_top_k,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
            trigger_mode=args.b_cto_trigger,
            trigger_threshold=trigger_threshold,
            embed_model_path=args.retrieval_embedding_model,
        )
        cto.save_result(
            output_path,
            orig_idx,
            item,
            completions,
            extra_fields={"b_cto_compute": b_cto_compute, "cto_compute": b_cto_compute},
        )
        logger.info(
            "Saved q=%s completions=%d neg_frac=%.3f processed=%d",
            orig_idx,
            len(completions),
            b_cto_compute["neg_branch_fraction"],
            b_cto_compute["processed_tokens_per_rollout_mean"],
        )

    logger.info("Done. Results in %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()
