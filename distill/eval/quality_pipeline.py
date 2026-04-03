"""
Generation, quality-gate filtering, and embedding phases for quality eval.

Provides:
    run_generation_phase(use_gguf, use_mlx, student, tokenizer, gguf_path, prompts, args) -> list[str]
    run_quality_gate_phase(prompts, instructions, responses) -> (samples, rejected, accumulators, lengths, categories)
    run_embedding_phase(use_mlx, use_gguf, student, tokenizer, samples, device, output_dir, batch_size) -> dict
"""

import logging
from pathlib import Path
from typing import Optional

from ..backends.mlx_utils import mlx_generate_responses
from ..backends.cpp_utils import generate_gguf_responses
from .quality_metrics import (
    MIN_RESPONSE_TOKENS,
    MAX_RESPONSE_TOKENS,
    TARGET_MIN_TOKENS,
    check_quality_gates,
    detect_category,
    diversity_metrics,
    compute_ngram_entropy,
)
from .quality_inference import (
    batch_generate_responses,
    compute_embedding_diversity,
    create_umap_visualization,
)

logger = logging.getLogger(__name__)


def run_generation_phase(
    use_gguf: bool,
    use_mlx: bool,
    student,
    tokenizer,
    gguf_path: Optional[str],
    prompts: list,
    args,
) -> list:
    """Generate student responses for all prompts.

    Returns:
        list of response strings, one per prompt
    """
    n = len(prompts)
    logger.info(
        "Generating %d responses with batch_size=%d...", n, args.batch_size
    )

    if use_gguf:
        responses = generate_gguf_responses(
            gguf_path,
            prompts,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            n_parallel=args.batch_size,
        )
    elif use_mlx:
        responses = mlx_generate_responses(
            student, tokenizer, prompts, args.max_new_tokens, args.temperature
        )
    else:
        responses = batch_generate_responses(
            student,
            tokenizer,
            prompts,
            args.max_new_tokens,
            args.temperature,
            args.device if hasattr(args, "device") else None,
            args.batch_size,
        )

    logger.info("Generated %d responses", len(responses))
    return responses


def run_quality_gate_phase(
    prompts: list,
    instructions: list,
    responses: list,
) -> tuple:
    """Apply quality gates, compute per-sample diversity, and categorize responses.

    Returns:
        (samples, rejected, diversity_accumulators, lengths, categories)
        where diversity_accumulators = {"d1_sum": float, "d2_sum": float, "max_rep_sum": int}
    """
    samples = []
    rejected = {"too_short": 0, "too_long": 0, "refusal": 0, "below_target": 0}
    d1_sum = d2_sum = 0.0
    max_rep_sum = 0
    lengths = []
    categories = []
    n = len(prompts)

    for i, (prompt, instruction, response) in enumerate(
        zip(prompts, instructions, responses)
    ):
        passed, reason, flags = check_quality_gates(response, instruction)

        for flag_key in flags:
            if flag_key in rejected:
                rejected[flag_key] += 1

        d1, d2, max_rep = diversity_metrics(response)
        d1_sum += d1
        d2_sum += d2
        max_rep_sum += max_rep
        length = len(response.split())
        lengths.append(length)

        category = detect_category(instruction, response)
        categories.append(category)

        sample = {
            "prompt": prompt,
            "instruction": instruction,
            "response": response,
            "distinct_1": round(d1, 4),
            "distinct_2": round(d2, 4),
            "max_rep": max_rep,
            "length_tokens": length,
            "category": category,
            "quality_gate_passed": passed,
            "quality_gate_reason": reason,
        }

        if passed:
            samples.append(sample)

        if (i + 1) % 20 == 0:
            logger.info(
                "  Processed %d/%d  avg_d1=%.3f  passed=%d  rejected=%d",
                i + 1,
                n,
                d1_sum / (i + 1),
                len(samples),
                i + 1 - len(samples),
            )

    # Summary logging
    n_passed = len(samples)
    n_rejected = n - n_passed
    refusal_rate = rejected["refusal"] / n * 100
    logger.info("")
    logger.info("Quality Gate Summary:")
    logger.info("  Passed: %d/%d (%.1f%%)", n_passed, n, n_passed / n * 100)
    logger.info("  Rejected: %d/%d (%.1f%%)", n_rejected, n, n_rejected / n * 100)
    logger.info("    - Too short (<%d tok): %d", MIN_RESPONSE_TOKENS, rejected["too_short"])
    logger.info("    - Too long (>%d tok): %d", MAX_RESPONSE_TOKENS, rejected["too_long"])
    logger.info("    - Refusals: %d (%.1f%%)", rejected["refusal"], refusal_rate)
    logger.info("    - Below target (<%d tok): %d", TARGET_MIN_TOKENS, rejected["below_target"])

    if refusal_rate > 5.0:
        logger.warning("Refusal rate %.1f%% exceeds 5%% threshold!", refusal_rate)

    category_counts = {cat: categories.count(cat) for cat in set(categories)}
    logger.info("")
    logger.info("Category Distribution:")
    for cat in sorted(category_counts.keys()):
        logger.info(
            "  %s: %d (%.1f%%)",
            cat,
            category_counts[cat],
            category_counts[cat] / len(categories) * 100,
        )

    avg_d1 = d1_sum / n
    avg_d2 = d2_sum / n
    avg_max_rep = max_rep_sum / n
    median_len = sorted(lengths)[n // 2]
    ngram_entropy = compute_ngram_entropy([s["response"] for s in samples], n=3)

    logger.info("")
    logger.info("Diversity Summary:")
    logger.info("  distinct-1: %.3f", avg_d1)
    logger.info("  distinct-2: %.3f", avg_d2)
    logger.info("  avg_max_rep: %.2f", avg_max_rep)
    logger.info("  3-gram entropy: %.2f bits", ngram_entropy)
    logger.info("  median_length: %d tokens", median_len)

    if avg_d1 < 0.5:
        logger.warning("Low distinct-1 (%.3f) — possible mode collapse", avg_d1)
    if avg_max_rep > 3:
        logger.warning(
            "High avg max repetition (%.1f) — check for repetition loops", avg_max_rep
        )

    accumulators = {
        "d1_sum": d1_sum,
        "d2_sum": d2_sum,
        "max_rep_sum": max_rep_sum,
        "ngram_entropy": ngram_entropy,
        "avg_d1": avg_d1,
        "avg_d2": avg_d2,
        "avg_max_rep": avg_max_rep,
        "median_len": median_len,
    }
    return samples, rejected, accumulators, lengths, categories


def run_embedding_phase(
    use_mlx: bool,
    use_gguf: bool,
    student,
    tokenizer,
    samples: list,
    device,
    output_dir: Path,
    batch_size: int,
) -> dict:
    """Compute embedding diversity and UMAP visualization (PyTorch only).

    Returns:
        dict with keys: enabled, mean_pairwise_distance, std_pairwise_distance, coverage_radius_95
    """
    if use_mlx or use_gguf:
        logger.info(
            "Skipping embedding diversity (not supported with MLX/GGUF backend)"
        )
        return {"enabled": False}

    logger.info("")
    logger.info("Computing embedding diversity...")
    try:
        emb_div = compute_embedding_diversity(
            student,
            tokenizer,
            [s["response"] for s in samples],
            device,
            batch_size=batch_size,
        )
        result = {
            "enabled": True,
            "mean_pairwise_distance": emb_div["mean_pairwise_distance"],
            "std_pairwise_distance": emb_div["std_pairwise_distance"],
            "coverage_radius_95": emb_div["coverage_radius_95"],
        }
        logger.info(
            "  Mean pairwise distance: %.4f", emb_div["mean_pairwise_distance"]
        )
        logger.info(
            "  Coverage radius (95%%): %.4f", emb_div["coverage_radius_95"]
        )

        umap_path = output_dir / "embedding_viz.json"
        create_umap_visualization(
            emb_div["embeddings"],
            [s["category"] for s in samples],
            umap_path,
        )
        return result
    except Exception as e:
        logger.error(
            "Failed to compute embedding diversity: %s", e, exc_info=True
        )
        return {"enabled": False}
