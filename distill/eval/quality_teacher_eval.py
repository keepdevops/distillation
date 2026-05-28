"""
Teacher model loading, LLM-as-judge scoring, and teacher perplexity for quality eval.

Provides:
    run_teacher_eval(args, use_gguf, use_mlx, samples, device, cache_dir, offline,
                     gguf_teacher_path) -> (judge_result, teacher_ppl_result)
"""

import logging
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..backends.mlx_utils import load_mlx_model, mlx_generate_responses, compute_mlx_perplexity
from ..backends.cpp_utils import find_gguf, generate_gguf_responses, compute_gguf_perplexity
from ..infra.config import cfg
from .quality_metrics import JUDGE_PROMPT, parse_judge_score
from .quality_inference import batch_judge_responses, compute_teacher_perplexity_on_responses

logger = logging.getLogger(__name__)


def _load_teacher(use_gguf, use_mlx, gguf_teacher_path, args, cache_dir, offline, device):
    """Internal: load teacher model/tokenizer for the chosen backend.

    Returns:
        (judge_model, judge_tok)  — both None for GGUF
    """
    if use_gguf and gguf_teacher_path:
        logger.info("GGUF teacher: %s", gguf_teacher_path)
        return None, None

    if use_mlx:
        judge_model, judge_tok = load_mlx_model(args.teacher)
        return judge_model, judge_tok

    try:
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.teacher,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=offline,
        )
        judge_tok = AutoTokenizer.from_pretrained(
            args.teacher,
            cache_dir=cache_dir,
            local_files_only=offline,
        )
        judge_tok.pad_token = judge_tok.eos_token
        judge_model.to(device)
        judge_model.eval()
        return judge_model, judge_tok
    except Exception as e:
        logger.error(
            "Failed to load teacher model %s: %s", args.teacher, e, exc_info=True
        )
        raise


def _run_teacher_perplexity(
    use_gguf, use_mlx, gguf_teacher_path, judge_model, judge_tok,
    samples, device, args
) -> dict:
    """Compute teacher perplexity on student outputs.

    Returns:
        teacher_ppl_result dict
    """
    logger.info("Computing teacher perplexity on student generations...")
    sample_prompts = [s["prompt"] for s in samples]
    sample_responses = [s["response"] for s in samples]
    avg_teacher_ppl = None

    try:
        if use_gguf and gguf_teacher_path:
            full_texts = [p + r for p, r in zip(sample_prompts, sample_responses)]
            raw_loss = compute_gguf_perplexity(gguf_teacher_path, full_texts, ctx_size=1024)
            avg_teacher_ppl = math.exp(min(raw_loss, 10)) if raw_loss else None
            for i in range(len(samples)):
                samples[i]["teacher_ppl"] = round(avg_teacher_ppl, 2) if avg_teacher_ppl else None

        elif use_mlx:
            full_texts = [p + r for p, r in zip(sample_prompts, sample_responses)]
            raw_losses = compute_mlx_perplexity(
                judge_model, judge_tok, full_texts, max_length=1024, batch_size=4
            )
            avg_teacher_ppl = math.exp(min(raw_losses, 10)) if raw_losses else None
            teacher_ppls = [round(avg_teacher_ppl, 2)] * len(samples) if avg_teacher_ppl else []
            for i in range(len(samples)):
                samples[i]["teacher_ppl"] = teacher_ppls[i] if teacher_ppls else None

        else:
            teacher_ppls = compute_teacher_perplexity_on_responses(
                judge_model, judge_tok, sample_prompts, sample_responses,
                device, batch_size=4
            )
            avg_teacher_ppl = sum(teacher_ppls) / len(teacher_ppls) if teacher_ppls else None
            for i, ppl in enumerate(teacher_ppls):
                samples[i]["teacher_ppl"] = round(ppl, 2)

    except Exception as e:
        logger.error("Teacher perplexity computation failed: %s", e, exc_info=True)
        return {"enabled": False}

    logger.info("  Avg teacher perplexity: %.2f", avg_teacher_ppl or 0)
    if avg_teacher_ppl and avg_teacher_ppl > 100:
        logger.warning(
            "High teacher perplexity (%.2f) on student outputs — check quality",
            avg_teacher_ppl,
        )

    return {
        "enabled": True,
        "avg_teacher_ppl": round(avg_teacher_ppl, 2) if avg_teacher_ppl else None,
    }


def _run_judge(
    use_gguf, use_mlx, gguf_teacher_path, judge_model, judge_tok,
    samples, device, args
) -> dict:
    """Run LLM-as-judge scoring on samples.

    Returns:
        judge_result dict
    """
    try:
        if use_gguf and gguf_teacher_path:
            logger.info(
                "GGUF judge: %d judgments via llama-server (parallel=%d)...",
                len(samples), args.batch_size,
            )
            judge_prompts = [
                JUDGE_PROMPT.format(instruction=s["instruction"], response=s["response"])
                for s in samples
            ]
            judgments = generate_gguf_responses(
                gguf_teacher_path, judge_prompts,
                max_tokens=60, temperature=0.0,
                n_parallel=args.batch_size,
                port=cfg.services.llama_teacher_port,
            )
        elif use_mlx:
            logger.info("MLX judge: generating %d judgments sequentially...", len(samples))
            judge_prompts = [
                JUDGE_PROMPT.format(instruction=s["instruction"], response=s["response"])
                for s in samples
            ]
            judgments = mlx_generate_responses(
                judge_model, judge_tok, judge_prompts, max_new_tokens=60, temperature=0.0
            )
        else:
            logger.info(
                "Batch judging %d responses (batch_size=%d)...", len(samples), args.batch_size
            )
            judgments = batch_judge_responses(
                judge_model, judge_tok,
                [s["instruction"] for s in samples],
                [s["response"] for s in samples],
                device, args.batch_size,
            )
    except Exception as e:
        logger.error("Judge generation failed: %s", e, exc_info=True)
        return {"enabled": False}

    scores = []
    for i, (s, raw) in enumerate(zip(samples, judgments)):
        score = parse_judge_score(raw)
        s["judge_raw"] = raw
        s["judge_score"] = score
        if score is not None:
            scores.append(score)
        if (i + 1) % 20 == 0:
            avg_so_far = sum(scores) / len(scores) if scores else float("nan")
            logger.info(
                "  Judged %d/%d  avg_score=%.2f", i + 1, len(samples), avg_so_far
            )

    valid_scores = [sc for sc in scores if sc is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
    logger.info(
        "Judge avg score: %.2f / 10 (%d/%d parseable)",
        avg_score or 0, len(valid_scores), len(samples),
    )
    if avg_score is not None and avg_score < 5:
        logger.warning(
            "Low judge avg score (%.2f) — check instruction corruption or mode collapse",
            avg_score,
        )

    return {
        "enabled": True,
        "teacher": args.teacher,
        "avg_score": round(avg_score, 2) if avg_score is not None else None,
        "n_scored": len(valid_scores),
        "scores": scores,
    }


def run_teacher_eval(
    args,
    use_gguf: bool,
    use_mlx: bool,
    samples: list,
    device,
    cache_dir: str,
    offline: bool,
    gguf_teacher_path,
) -> tuple:
    """Load teacher and run judge + perplexity evaluation on samples.

    Returns:
        (judge_result, teacher_ppl_result)
    """
    judge_result = {"enabled": False}
    teacher_ppl_result = {"enabled": False}

    if not (args.judge or args.judge_teacher_ppl):
        return judge_result, teacher_ppl_result

    logger.info("")
    logger.info("Loading teacher: %s", args.teacher)
    judge_model, judge_tok = _load_teacher(
        use_gguf, use_mlx, gguf_teacher_path, args, cache_dir, offline, device
    )

    if args.judge_teacher_ppl:
        teacher_ppl_result = _run_teacher_perplexity(
            use_gguf, use_mlx, gguf_teacher_path,
            judge_model, judge_tok, samples, device, args,
        )

    if args.judge:
        judge_result = _run_judge(
            use_gguf, use_mlx, gguf_teacher_path,
            judge_model, judge_tok, samples, device, args,
        )

    # Cleanup
    if judge_model is not None:
        del judge_model
    if use_mlx:
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception as e:
            logger.error("Failed to clear MLX cache: %s", e, exc_info=True)
    elif not use_gguf and device.type == "mps":
        torch.mps.empty_cache()

    return judge_result, teacher_ppl_result
