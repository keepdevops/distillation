#!/usr/bin/env python3
"""
Generation-based quality eval: diversity metrics + optional LLM-as-judge.

Samples N prompts from the validation split, generates student responses,
computes distinct-1/distinct-2/max-repetition, and optionally scores each
response with a teacher model (LLM-as-judge).

Output: {output_dir}/quality_metrics.json

Usage:
    python -m distill.eval_quality ./distilled-minillm
    python -m distill.eval_quality ./distilled-minillm --judge
    python -m distill.eval_quality ./distilled-minillm --judge --teacher Qwen/Qwen2-1.5B-Instruct
    python -m distill.eval_quality ./distilled-minillm --checkpoint ./distilled-minillm/checkpoint-80
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..data.pipeline import load_dataset_split, format_prompt_only
from ..infra.train_utils import get_device
from ..backends.cpp_utils import find_gguf

from .quality_backend import select_backend, load_student_backend
from .quality_pipeline import run_generation_phase, run_quality_gate_phase, run_embedding_phase
from .quality_teacher_eval import run_teacher_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

    # Backend selection
    use_gguf, use_mlx, gguf_student_path = select_backend(args, checkpoint_dir)

    # Load student
    student, tokenizer = load_student_backend(
        use_gguf, use_mlx, checkpoint_dir, args, cache_dir, offline, device
    )

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
    prompts = [ex["prompt"] for ex in val_ds]
    instructions = [ex.get("instruction", ex["prompt"]) for ex in val_ds]

    # Phase 1: Generation
    responses = run_generation_phase(
        use_gguf, use_mlx, student, tokenizer, gguf_student_path, prompts, args
    )

    # Phase 2: Quality gate + diversity
    samples, rejected, accumulators, lengths, categories = run_quality_gate_phase(
        prompts, instructions, responses
    )

    # Phase 3: Embedding diversity (before loading teacher)
    embedding_result = run_embedding_phase(
        use_mlx, use_gguf, student, tokenizer, samples, device, output_dir, args.batch_size
    )

    # Free student before loading teacher
    if student is not None:
        del student
    if not use_gguf and not use_mlx and device.type == "mps":
        torch.mps.empty_cache()

    # Phase 4: Teacher eval (judge + perplexity)
    gguf_teacher_path = find_gguf(args.teacher) if use_gguf else None
    judge_result, teacher_ppl_result = run_teacher_eval(
        args, use_gguf, use_mlx, samples, device, cache_dir, offline, gguf_teacher_path
    )

    # Assemble result
    n_passed = len(samples)
    n_rejected = n - n_passed
    refusal_rate = rejected["refusal"] / n * 100
    category_counts = {cat: categories.count(cat) for cat in set(categories)}
    category_pcts = {cat: count / len(categories) * 100 for cat, count in category_counts.items()}

    result = {
        "model_dir": str(output_dir),
        "checkpoint": str(checkpoint_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples_generated": n,
        "n_samples_passed": len(samples),
        "quality_gates": {
            "passed": n_passed,
            "rejected": n_rejected,
            "pass_rate_pct": round(n_passed / n * 100, 1),
            "refusal_rate_pct": round(refusal_rate, 2),
            "rejection_reasons": rejected,
        },
        "category_distribution": {
            "counts": category_counts,
            "percentages": {cat: round(pct, 1) for cat, pct in category_pcts.items()},
        },
        "diversity": {
            "avg_distinct_1": round(accumulators["avg_d1"], 4),
            "avg_distinct_2": round(accumulators["avg_d2"], 4),
            "avg_max_rep": round(accumulators["avg_max_rep"], 2),
            "ngram_entropy_3": round(accumulators["ngram_entropy"], 2),
            "median_response_tokens": accumulators["median_len"],
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
