#!/usr/bin/env python3
"""
Generation-based quality eval: diversity metrics + optional LLM-as-judge.

Samples N prompts from the validation split, generates student responses,
computes distinct-1/distinct-2/max-repetition, and optionally scores each
response with a teacher model (LLM-as-judge).

Output: {output_dir}/quality_metrics.json

Usage:
    python scripts/eval_quality.py ./distilled-minillm
    python scripts/eval_quality.py ./distilled-minillm --judge
    python scripts/eval_quality.py ./distilled-minillm --judge --teacher Qwen/Qwen2-1.5B-Instruct
    python scripts/eval_quality.py ./distilled-minillm --checkpoint ./distilled-minillm/checkpoint-80
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from transformers import GenerationConfig

# Add scripts directory to path for local imports
from ..data.pipeline import load_dataset_split, format_prompt_only, REFUSAL_PATTERNS, is_refusal
from ..infra.train_utils import get_device, load_student_model
from ..backends.mlx_utils import is_mlx_available, load_mlx_model, mlx_generate_responses, compute_mlx_perplexity
from ..backends.cpp_utils import is_cpp_available, find_gguf, generate_gguf_responses, compute_gguf_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Production quality gates

from .quality_metrics import (
    MIN_RESPONSE_TOKENS, MAX_RESPONSE_TOKENS, TARGET_MIN_TOKENS, JUDGE_PROMPT,
    check_length_valid, check_quality_gates, detect_category,
    diversity_metrics, compute_ngram_entropy, parse_judge_score,
)
from .quality_inference import (
    batch_generate_responses, compute_embedding_diversity,
    create_umap_visualization, batch_judge_responses,
    compute_teacher_perplexity_on_responses,
)

OPEN_STUDENT = "Qwen/Qwen2-0.5B-Instruct"
OPEN_TEACHER = "Qwen/Qwen2-1.5B-Instruct"


def parse_args():
    from ..infra.cli_common import add_cache_and_offline
    p = argparse.ArgumentParser(description="Generation-based quality eval: diversity + judge")
    p.add_argument("output_dir", type=str, help="Training output dir")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Model checkpoint dir (default: output_dir)")
    p.add_argument("--student", type=str, default=OPEN_STUDENT)
    p.add_argument("--teacher", type=str, default=OPEN_TEACHER,
                   help="Teacher model for LLM-as-judge and perplexity scoring")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--val_size", type=float, default=0.02)
    p.add_argument("--n_samples", type=int, default=100,
                   help="Number of samples to generate responses for (default: 100)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--backend", type=str, default="auto",
                   choices=["auto", "pytorch", "mlx", "gguf"])
    p.add_argument("--judge", action="store_true",
                   help="Run LLM-as-judge scoring")
    p.add_argument("--judge_teacher_ppl", action="store_true",
                   help="Compute teacher perplexity on student outputs")
    add_cache_and_offline(p)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else output_dir
    out_path = output_dir / "quality_metrics.json"

    offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    cache_dir = os.environ.get("HF_HOME") or args.cache_dir
    device = get_device()

    # Backend selection: gguf (C++/Metal) > mlx > pytorch
    use_gguf = False
    use_mlx = False

    if args.backend == "gguf":
        if not is_cpp_available():
            logger.warning("GGUF backend requested but llama.cpp binaries not found, falling back")
        else:
            use_gguf = True
            logger.info("Backend: GGUF/llama.cpp (Metal, parallel generation)")
    elif args.backend == "mlx":
        if not is_mlx_available():
            logger.warning("MLX backend requested but mlx/mlx-lm not installed, falling back to PyTorch")
        else:
            use_mlx = True
            logger.info("Backend: MLX (Apple Silicon optimized)")
    elif args.backend == "auto":
        gguf_candidate = find_gguf(str(checkpoint_dir))
        if gguf_candidate and is_cpp_available():
            use_gguf = True
            logger.info("Backend: GGUF/llama.cpp (auto-detected: %s)", Path(gguf_candidate).name)
        elif is_mlx_available():
            use_mlx = True
            logger.info("Backend: MLX (auto-detected)")
        else:
            logger.info("Backend: PyTorch, Device: %s", device)
    else:
        logger.info("Backend: PyTorch, Device: %s", device)

    # Load student
    logger.info("Loading student from %s", checkpoint_dir)
    gguf_student_path = None
    if use_gguf:
        gguf_student_path = find_gguf(str(checkpoint_dir))
        if gguf_student_path is None:
            logger.warning("No .gguf found in %s — falling back to MLX/PyTorch", checkpoint_dir)
            use_gguf = False
            use_mlx = is_mlx_available()

    if use_gguf:
        student, tokenizer = None, None   # generation handled via llama-server subprocess
        logger.info("GGUF student: %s", gguf_student_path)
    elif use_mlx:
        student, tokenizer = load_mlx_model(checkpoint_dir, student_name=args.student)
    else:
        student, tokenizer = load_student_model(checkpoint_dir, args.student, cache_dir, offline, device)

    # Load dataset
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    logger.info("Loading dataset: %s", args.dataset)
    dataset = load_dataset_split(args.dataset, args.max_samples, ds_cache, offline)
    dataset = dataset.map(
        lambda ex: {
            "prompt": format_prompt_only(ex),
            "instruction": ex.get("instruction", "").strip(),
        },
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=args.val_size, seed=42)
    val_ds = split["test"]

    n = min(args.n_samples, len(val_ds))
    val_ds = val_ds.select(range(n))
    logger.info("Generating %d responses with batch_size=%d...", n, args.batch_size)

    # Phase 1: Batch generation
    prompts = [ex["prompt"] for ex in val_ds]
    instructions = [ex.get("instruction", ex["prompt"]) for ex in val_ds]

    if use_gguf:
        responses = generate_gguf_responses(
            gguf_student_path, prompts,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            n_parallel=args.batch_size,   # reuse batch_size as server parallelism
        )
    elif use_mlx:
        responses = mlx_generate_responses(
            student, tokenizer, prompts, args.max_new_tokens, args.temperature
        )
    else:
        responses = batch_generate_responses(
            student, tokenizer, prompts, args.max_new_tokens,
            args.temperature, device, args.batch_size
        )
    logger.info("Generated %d responses", len(responses))

    # Phase 2: Quality gate filtering
    samples = []
    rejected = {"too_short": 0, "too_long": 0, "refusal": 0, "below_target": 0}
    d1_sum = d2_sum = 0.0
    max_rep_sum = 0
    lengths = []
    categories = []

    for i, (prompt, instruction, response) in enumerate(zip(prompts, instructions, responses)):
        # Quality gates
        passed, reason, flags = check_quality_gates(response, instruction)

        # Track all rejections
        for flag_key in flags:
            if flag_key in rejected:
                rejected[flag_key] += 1

        # Diversity metrics
        d1, d2, max_rep = diversity_metrics(response)
        d1_sum += d1
        d2_sum += d2
        max_rep_sum += max_rep
        length = len(response.split())
        lengths.append(length)

        # Category detection
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

        # Only include in final samples if passed quality gates
        if passed:
            samples.append(sample)

        if (i + 1) % 20 == 0:
            logger.info("  Processed %d/%d  avg_d1=%.3f  passed=%d  rejected=%d",
                       i + 1, n, d1_sum / (i + 1), len(samples), i + 1 - len(samples))

    # Quality gate summary
    n_passed = len(samples)
    n_rejected = n - n_passed
    refusal_rate = rejected["refusal"] / n * 100
    logger.info("")
    logger.info("Quality Gate Summary:")
    logger.info("  Passed: %d/%d (%.1f%%)", n_passed, n, n_passed/n*100)
    logger.info("  Rejected: %d/%d (%.1f%%)", n_rejected, n, n_rejected/n*100)
    logger.info("    - Too short (<%d tok): %d", MIN_RESPONSE_TOKENS, rejected["too_short"])
    logger.info("    - Too long (>%d tok): %d", MAX_RESPONSE_TOKENS, rejected["too_long"])
    logger.info("    - Refusals: %d (%.1f%%)", rejected["refusal"], refusal_rate)
    logger.info("    - Below target (<%d tok): %d", TARGET_MIN_TOKENS, rejected["below_target"])

    if refusal_rate > 5.0:
        logger.warning("Refusal rate %.1f%% exceeds 5%% threshold!", refusal_rate)

    # Category balance
    category_counts = {cat: categories.count(cat) for cat in set(categories)}
    category_pcts = {cat: count/len(categories)*100 for cat, count in category_counts.items()}
    logger.info("")
    logger.info("Category Distribution:")
    for cat in sorted(category_counts.keys()):
        logger.info("  %s: %d (%.1f%%)", cat, category_counts[cat], category_pcts[cat])

    # Diversity summary
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
        logger.warning("High avg max repetition (%.1f) — check for repetition loops", avg_max_rep)

    # Phase 3: Embedding diversity (compute before loading teacher)
    embedding_result = {"enabled": False}
    if not use_mlx and not use_gguf:
        logger.info("")
        logger.info("Computing embedding diversity...")
        try:
            emb_div = compute_embedding_diversity(
                student, tokenizer,
                [s["response"] for s in samples],
                device, batch_size=args.batch_size
            )
            embedding_result = {
                "enabled": True,
                "mean_pairwise_distance": emb_div["mean_pairwise_distance"],
                "std_pairwise_distance": emb_div["std_pairwise_distance"],
                "coverage_radius_95": emb_div["coverage_radius_95"],
            }
            logger.info("  Mean pairwise distance: %.4f", emb_div["mean_pairwise_distance"])
            logger.info("  Coverage radius (95%%): %.4f", emb_div["coverage_radius_95"])

            # UMAP visualization
            umap_path = output_dir / "embedding_viz.json"
            create_umap_visualization(
                emb_div["embeddings"],
                [s["category"] for s in samples],
                umap_path
            )
        except Exception as e:
            logger.warning("Failed to compute embedding diversity: %s", e)
    else:
        logger.info("Skipping embedding diversity (not supported with MLX/GGUF backend)")

    # Phase 4: Teacher evaluation (judge + perplexity)
    if student is not None:
        del student
    if not use_gguf and not use_mlx and device.type == "mps":
        torch.mps.empty_cache()

    judge_result = {"enabled": False}
    teacher_ppl_result = {"enabled": False}

    if args.judge or args.judge_teacher_ppl:
        logger.info("")
        logger.info("Loading teacher: %s", args.teacher)

        gguf_teacher_path = find_gguf(args.teacher) if use_gguf else None
        if use_gguf and gguf_teacher_path:
            judge_model, judge_tok = None, None   # teacher gen via llama-server
            logger.info("GGUF teacher: %s", gguf_teacher_path)
        elif use_mlx:
            judge_model, judge_tok = load_mlx_model(args.teacher)
        else:
            judge_model = AutoModelForCausalLM.from_pretrained(
                args.teacher, dtype=torch.bfloat16, device_map="auto",
                cache_dir=cache_dir, local_files_only=offline,
            )
            judge_tok = AutoTokenizer.from_pretrained(
                args.teacher, cache_dir=cache_dir, local_files_only=offline,
            )
            judge_tok.pad_token = judge_tok.eos_token
            judge_model.to(device)
            judge_model.eval()

        # Teacher perplexity on student outputs
        if args.judge_teacher_ppl:
            logger.info("Computing teacher perplexity on student generations...")
            sample_prompts = [s["prompt"] for s in samples]
            sample_responses = [s["response"] for s in samples]

            if use_gguf and gguf_teacher_path:
                full_texts = [p + r for p, r in zip(sample_prompts, sample_responses)]
                raw_loss = compute_gguf_perplexity(gguf_teacher_path, full_texts, ctx_size=1024)
                avg_teacher_ppl = math.exp(min(raw_loss, 10)) if raw_loss else None
                for i in range(len(samples)):
                    samples[i]["teacher_ppl"] = round(avg_teacher_ppl, 2) if avg_teacher_ppl else None
            elif use_mlx:
                full_texts = [p + r for p, r in zip(sample_prompts, sample_responses)]
                raw_losses = compute_mlx_perplexity(judge_model, judge_tok, full_texts, max_length=1024, batch_size=4)
                # compute_mlx_perplexity returns mean loss; approximate per-sample as uniform
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

            teacher_ppl_result = {
                "enabled": True,
                "avg_teacher_ppl": round(avg_teacher_ppl, 2) if avg_teacher_ppl else None,
            }
            logger.info("  Avg teacher perplexity: %.2f", avg_teacher_ppl or 0)

            if avg_teacher_ppl and avg_teacher_ppl > 100:
                logger.warning("High teacher perplexity (%.2f) on student outputs — check quality", avg_teacher_ppl)

        # LLM-as-judge scoring (PyTorch only — requires model.generate)
        if args.judge:
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
                    port=8090,  # use different port from student server
                )
            elif use_mlx:
                logger.info("MLX judge: generating %d judgments sequentially...", len(samples))
                sample_instructions = [s["instruction"] for s in samples]
                sample_responses_list = [s["response"] for s in samples]
                judge_prompts = [
                    JUDGE_PROMPT.format(instruction=inst, response=resp)
                    for inst, resp in zip(sample_instructions, sample_responses_list)
                ]
                judgments = mlx_generate_responses(judge_model, judge_tok, judge_prompts, max_new_tokens=60, temperature=0.0)
            else:
                logger.info("Batch judging %d responses (batch_size=%d)...", len(samples), args.batch_size)
                sample_instructions = [s["instruction"] for s in samples]
                sample_responses_list = [s["response"] for s in samples]
                judgments = batch_judge_responses(
                    judge_model, judge_tok, sample_instructions, sample_responses_list,
                    device, args.batch_size
                )

            scores = []
            for i, (s, raw) in enumerate(zip(samples, judgments)):
                score = parse_judge_score(raw)
                s["judge_raw"] = raw
                s["judge_score"] = score
                if score is not None:
                    scores.append(score)

                if (i + 1) % 20 == 0:
                    avg_so_far = sum(scores) / len(scores) if scores else float("nan")
                    logger.info("  Judged %d/%d  avg_score=%.2f", i + 1, len(samples), avg_so_far)

            valid_scores = [sc for sc in scores if sc is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            logger.info("Judge avg score: %.2f / 10 (%d/%d parseable)", avg_score or 0, len(valid_scores), len(samples))

            if avg_score is not None and avg_score < 5:
                logger.warning("Low judge avg score (%.2f) — check instruction corruption or mode collapse",
                               avg_score)

            judge_result = {
                "enabled": True,
                "teacher": args.teacher,
                "avg_score": round(avg_score, 2) if avg_score is not None else None,
                "n_scored": len(valid_scores),
                "scores": scores,
            }

        if judge_model is not None:
            del judge_model
        if use_mlx:
            import mlx.core as mx
            mx.clear_cache()
        elif not use_gguf and device.type == "mps":
            torch.mps.empty_cache()

    result = {
        "model_dir": str(output_dir),
        "checkpoint": str(checkpoint_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples_generated": n,
        "n_samples_passed": len(samples),
        "quality_gates": {
            "passed": n_passed,
            "rejected": n_rejected,
            "pass_rate_pct": round(n_passed/n*100, 1),
            "refusal_rate_pct": round(refusal_rate, 2),
            "rejection_reasons": rejected,
        },
        "category_distribution": {
            "counts": category_counts,
            "percentages": {cat: round(pct, 1) for cat, pct in category_pcts.items()},
        },
        "diversity": {
            "avg_distinct_1": round(avg_d1, 4),
            "avg_distinct_2": round(avg_d2, 4),
            "avg_max_rep": round(avg_max_rep, 2),
            "ngram_entropy_3": round(ngram_entropy, 2),
            "median_response_tokens": median_len,
        },
        "embedding_diversity": embedding_result,
        "teacher_perplexity": teacher_ppl_result,
        "judge": judge_result,
        "samples": samples,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Quality metrics saved to %s", out_path)


if __name__ == "__main__":
    main()
