"""
Argument parser and logging setup for the autonomous distillation agent.
"""
from __future__ import annotations

import argparse
import logging


def build_arg_parser() -> argparse.ArgumentParser:
    """Return a fully configured ArgumentParser for the distillation agent."""
    ap = argparse.ArgumentParser(description="Autonomous distillation agent")
    ap.add_argument("--config", type=str, help="JSON config (overrides CLI)")
    ap.add_argument("--output_dir", type=str, default="./distilled-minillm")
    ap.add_argument("--open", action="store_true",
                    help="Use Qwen2 open models (no HF login)")
    ap.add_argument("--offline", action="store_true",
                    help="Air-gapped: local cache only")
    ap.add_argument("--watchdog", action="store_true",
                    help="Enable pause.flag / plateau detection")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last epoch checkpoint in output_dir (MLX only)")
    ap.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        choices=["pytorch", "mlx", "unsloth"],
        help="Training backend (default: pytorch — existing behavior unchanged)",
    )
    ap.add_argument(
        "--export",
        type=str,
        default="gguf",
        choices=["gguf", "coreml", "mlx", "all", "none"],
        help="Export format after distillation (default: gguf)",
    )
    # Legacy alias kept for backwards compat
    ap.add_argument("--export-gguf", action="store_true",
                    help="[legacy] Equivalent to --export gguf")
    ap.add_argument("--outtype", type=str, default="f16",
                    help="GGUF quantization type (f16, q8_0, q4_K_M)")
    ap.add_argument("--q_bits", type=int, default=4, choices=[4, 8],
                    help="MLX quantization bits (4 or 8)")
    ap.add_argument("--coreml_quantize", type=str, default=None,
                    choices=["int4", "int8", "float16"],
                    help="CoreML post-training quantization type")
    ap.add_argument("--dataset", type=str, default="tatsu-lab/alpaca",
                    help="HF dataset ID or local path (default: tatsu-lab/alpaca). "
                         "Better options: HuggingFaceH4/no_robots, teknium/OpenHermes-2.5, "
                         "allenai/tulu-3-sft-mixture, Open-Orca/OpenOrca")
    # ── Dataset filtering ─────────────────────────────────────────────────────
    ap.add_argument("--filter", action="store_true",
                    help="Pre-filter dataset with filter_dataset.py before distillation")
    ap.add_argument("--filter_target", type=int, default=10000,
                    help="Keep top-N samples after filtering (default: 10000)")
    ap.add_argument("--filter_min_response_words", type=int, default=30,
                    help="Min response word count for filter (default: 30)")
    ap.add_argument("--filter_min_distinct2", type=float, default=0.40,
                    help="Min distinct-2 score for filter (default: 0.40)")
    ap.add_argument("--filter_jaccard", type=float, default=0.55,
                    help="Jaccard similarity threshold for near-dedup (default: 0.55)")
    ap.add_argument("--filter_minhash", action="store_true",
                    help="Use MinHash LSH for global dedup in filter (requires datasketch)")
    ap.add_argument("--filter_teacher_score", action="store_true",
                    help="Re-rank filtered candidates by teacher log-probability")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=2,
                    help="Per-device batch size (default: 2)")
    ap.add_argument("--grad_acc", type=int, default=4,
                    help="Gradient accumulation steps (default: 4)")
    ap.add_argument("--learning_rate", type=float, default=2e-4,
                    help="Learning rate (default: 2e-4)")
    ap.add_argument("--ce_alpha", type=float, default=0.1,
                    help="CE loss weight mixed with KD loss (default: 0.1)")
    ap.add_argument("--multi_turn_ratio", type=float, default=0.0,
                    help="Fraction of samples formatted as multi-turn ChatML (default: 0.0)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="KD temperature (default 1.0)")
    ap.add_argument("--lora_r", type=int, default=16,
                    help="LoRA rank (default: 16). Higher ranks increase memory; "
                         "64 risks OOM on MLX with full-vocab logit tensors.")
    ap.add_argument("--eval_steps", type=int, default=20,
                    help="Eval frequency in gradient steps for pytorch backend (default: 20)")
    ap.add_argument("--num_generations", type=int, default=4,
                    help="GRPO completions per prompt for pytorch backend "
                         "(default: 4; fewer → frac_reward_zero_std high → no gradient)")
    ap.add_argument("--max_new_tokens", type=int, default=256,
                    help="Max generation length for pytorch backend "
                         "(default: 256; paired with _MAX_NATURAL_CHARS=800 in distill_minillm.py)")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--skip_eval", action="store_true",
                    help="Skip post-distillation eval steps (run_eval + eval_quality)")
    ap.add_argument("--compare_teacher", action="store_true",
                    help="Include teacher perplexity comparison in eval (loads teacher model)")
    ap.add_argument("--skip_judge", action="store_true",
                    help="Skip LLM-as-judge quality eval (eval_quality --judge)")
    # ── Curriculum distillation ───────────────────────────────────────────────
    ap.add_argument("--curriculum", action="store_true",
                    help="Run SFT warmup (distill_sft.py) before KD stage (pytorch only)")
    ap.add_argument("--sft_epochs", type=int, default=1,
                    help="SFT warmup epochs (default: 1)")
    # ── Synthetic data generation ─────────────────────────────────────────────
    ap.add_argument("--synthetic_data", action="store_true",
                    help="Generate synthetic data before distillation")
    ap.add_argument("--n_synthetic", type=int, default=2000,
                    help="Number of synthetic pairs to generate (default: 2000)")
    # ── Benchmarks ────────────────────────────────────────────────────────────
    ap.add_argument("--benchmarks", action="store_true",
                    help="Run WikiText-2 perplexity benchmark after eval")
    ap.add_argument("--baseline_dir", type=str, default=None,
                    help="Previous run dir for regression detection in benchmarks")
    # ── Experiment log ────────────────────────────────────────────────────────
    ap.add_argument("--log_experiment", action="store_true", default=True,
                    help="Log run to experiment_log.jsonl (default: on)")
    ap.add_argument("--no_log_experiment", dest="log_experiment", action="store_false",
                    help="Disable experiment logging")
    ap.add_argument("--experiment_log", type=str, default="experiment_log.jsonl",
                    help="Path to experiment log file")
    # ── Resource check ────────────────────────────────────────────────────────
    ap.add_argument("--memory_check", action="store_true",
                    help="Estimate memory usage before starting and warn if near 36GB")
    # ── Multi-trial search ────────────────────────────────────────────────────
    ap.add_argument("--n_trials", type=int, default=1,
                    help="Run N independent trials; best by eval_perplexity wins (default: 1)")
    # ── Reproducibility ──────────────────────────────────────────────────────
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed passed to all training scripts (default: 42)")
    # ── Auto naming ──────────────────────────────────────────────────────────
    ap.add_argument("--auto_name", action="store_true",
                    help="Auto-generate output_dir name from config hash + timestamp")
    return ap


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
