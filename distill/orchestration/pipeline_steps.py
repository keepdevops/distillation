"""
Individual pipeline step functions for the autonomous distillation agent.

Each function encapsulates one optional stage (filter, synth, sft warmup,
eval, quality, benchmark) and calls run_cmd internally.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .subprocess_runner import run_cmd

LOG = logging.getLogger(__name__)


def run_filter_step(
    args,
    output_dir: Path,
    project_root: Path,
) -> str:
    """Pre-filter the dataset and return the path to filtered output dir."""
    _filter_out = output_dir / "filtered_data"
    LOG.info(
        "Pre-filtering dataset '%s' → top-%d samples ...",
        args.dataset, args.filter_target,
    )
    filter_cmd = [
        sys.executable, "-m", "distill.data.filter",
        "--dataset", args.dataset,
        "--output_dir", str(_filter_out),
        "--target", str(args.filter_target),
        "--min_response_words", str(args.filter_min_response_words),
        "--min_distinct2", str(args.filter_min_distinct2),
        "--jaccard_threshold", str(args.filter_jaccard),
    ]
    if args.filter_minhash:
        filter_cmd.append("--minhash")
    if args.filter_teacher_score:
        filter_cmd += ["--teacher_score"]
        if args.open:
            filter_cmd += ["--teacher", "Qwen/Qwen2-1.5B-Instruct"]
    if args.offline:
        filter_cmd.append("--offline")
    run_cmd(filter_cmd, project_root)
    dataset_override = str(_filter_out)
    LOG.info("Filtered dataset ready: %s", dataset_override)
    return dataset_override


def run_synthetic_step(
    args,
    output_dir: Path,
    project_root: Path,
) -> str:
    """Generate synthetic data and return the path to the output dir."""
    LOG.info("Generating %d synthetic pairs...", args.n_synthetic)
    synth_cmd = [
        sys.executable, "-m", "distill.data.synth",
        "--output_dir", str(output_dir),
        "--n_generate", str(args.n_synthetic),
    ]
    if args.open:
        synth_cmd.append("--open")
    if args.offline:
        synth_cmd.append("--offline")
    run_cmd(synth_cmd, project_root)
    dataset_override = str(output_dir / "synthetic_data")
    LOG.info("Synthetic dataset ready: %s", dataset_override)
    return dataset_override


def run_sft_warmup(
    args,
    output_dir: Path,
    project_root: Path,
    dataset_override: str | None,
) -> str:
    """Run SFT curriculum warmup and return the checkpoint path."""
    LOG.info("Running SFT warmup (%d epoch(s))...", args.sft_epochs)
    sft_cmd = [
        sys.executable, "-m", "distill.training.backends.sft",
        "--output_dir", str(output_dir),
        "--epochs", str(args.sft_epochs),
        "--max_samples", str(args.max_samples),
        "--lora_r", str(args.lora_r),
    ]
    if args.open:
        sft_cmd.append("--open")
    if args.offline:
        sft_cmd.append("--offline")
    if args.watchdog:
        sft_cmd.append("--watchdog")
    if dataset_override:
        sft_cmd += ["--dataset", dataset_override]
    run_cmd(sft_cmd, project_root)
    sft_checkpoint = str(output_dir / "sft_checkpoint")
    LOG.info("SFT checkpoint ready: %s", sft_checkpoint)
    return sft_checkpoint


def run_eval_step(
    args,
    output_dir: Path,
    project_root: Path,
) -> None:
    """Run perplexity eval (student vs teacher vs quant).

    Skipped automatically for MLX backend since weights are not in HF format.
    """
    if args.backend == "mlx":
        LOG.info("Skipping perplexity eval for MLX backend (weights not in HF format).")
        return
    LOG.info("Running perplexity eval...")
    eval_cmd = [sys.executable, "-m", "distill.eval.perplexity", str(output_dir)]
    if args.open:
        eval_cmd += ["--student", "Qwen/Qwen2-0.5B-Instruct"]
    if args.offline:
        eval_cmd += ["--offline"]
    if args.compare_teacher:
        eval_cmd += ["--compare_teacher"]
        if args.open:
            eval_cmd += ["--teacher", "Qwen/Qwen2-1.5B-Instruct"]
    mlx_quant_dir = output_dir / f"mlx_q{args.q_bits}"
    if mlx_quant_dir.exists():
        eval_cmd += ["--quant_dir", str(mlx_quant_dir)]
        LOG.info("Quant dir found: %s", mlx_quant_dir)
    run_cmd(eval_cmd, project_root)


def run_quality_step(
    args,
    output_dir: Path,
    project_root: Path,
) -> None:
    """Run diversity + LLM-as-judge quality eval on the winning model."""
    LOG.info("Running quality eval (diversity + judge) on winning model...")
    quality_cmd = [sys.executable, "-m", "distill.eval.quality", str(output_dir)]
    if args.open:
        quality_cmd += ["--student", "Qwen/Qwen2-0.5B-Instruct"]
    if args.offline:
        quality_cmd += ["--offline"]
    if args.backend == "mlx":
        quality_cmd += ["--backend", "mlx"]
    if args.compare_teacher:
        quality_cmd += ["--judge"]
        if args.open:
            quality_cmd += ["--teacher", "Qwen/Qwen2-1.5B-Instruct"]
    run_cmd(quality_cmd, project_root)


def run_benchmark_step(
    args,
    output_dir: Path,
    project_root: Path,
) -> None:
    """Run WikiText-2 perplexity benchmark."""
    LOG.info("Running WikiText-2 benchmark...")
    bench_cmd = [sys.executable, "-m", "distill.eval.benchmarks", str(output_dir)]
    if args.open:
        bench_cmd += ["--student", "Qwen/Qwen2-0.5B-Instruct"]
    if args.offline:
        bench_cmd += ["--offline"]
    if args.baseline_dir:
        bench_cmd += ["--baseline_dir", args.baseline_dir]
    run_cmd(bench_cmd, project_root)
