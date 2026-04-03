"""
Build distillation subprocess commands for each training backend.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)


def _build_distill_cmd(
    args,
    output_dir: Path,
    sft_checkpoint: str | None,
    dataset_override: str | None,
    trial_seed: int,
) -> list[str]:
    """Return the distillation subprocess command for the given config."""
    if args.backend == "pytorch":
        cmd = [
            sys.executable, "-m", "distill.training.backends.minillm",
            "--output_dir", str(output_dir),
            "--epochs", str(args.epochs),
            "--max_samples", str(args.max_samples),
            "--minillm_temp", str(args.temperature),
            "--lora_r", str(args.lora_r),
            "--seed", str(trial_seed),
            "--batch_size", str(getattr(args, "batch_size", 8)),
            "--grad_acc", str(getattr(args, "grad_acc", 8)),
            "--learning_rate", str(getattr(args, "learning_rate", 2e-5)),
            "--eval_steps", str(getattr(args, "eval_steps", 20)),
            "--num_generations", str(getattr(args, "num_generations", 4)),
            "--max_new_tokens", str(getattr(args, "max_new_tokens", 256)),
        ]
        if args.open:
            cmd.append("--open")
        if args.offline:
            cmd.append("--offline")
        if args.watchdog:
            cmd.append("--watchdog")
        if sft_checkpoint:
            cmd += ["--student", sft_checkpoint]
        _ds = dataset_override or getattr(args, "dataset", None)
        if _ds:
            cmd += ["--dataset", _ds]

    elif args.backend == "mlx":
        cmd = [
            sys.executable, "-m", "distill.training.backends.mlx",
            "--output_dir", str(output_dir),
            "--epochs", str(args.epochs),
            "--max_samples", str(args.max_samples),
            "--kd_temp", str(args.temperature),
            "--lora_r", str(args.lora_r),
            "--q_bits", str(args.q_bits),
            "--seed", str(trial_seed),
            "--batch_size", str(getattr(args, "batch_size", 2)),
            "--grad_acc", str(getattr(args, "grad_acc", 8)),
            "--learning_rate", str(getattr(args, "learning_rate", 2e-4)),
            "--ce_alpha", str(getattr(args, "ce_alpha", 0.2)),
            "--multi_turn_ratio", str(getattr(args, "multi_turn_ratio", 0.0)),
            "--eval_steps", str(getattr(args, "eval_steps", 50)),
            "--topk_logits", str(getattr(args, "topk_logits", 50)),
        ]
        _ds = dataset_override or getattr(args, "dataset", None)
        if _ds:
            cmd += ["--dataset", _ds]
        if args.open:
            cmd.append("--open")
        if args.offline:
            cmd.append("--offline")
        if args.watchdog:
            cmd.append("--watchdog")
        if args.export not in ("mlx", "all"):
            cmd.append("--no_export")
        if getattr(args, "resume", False):
            cmd.append("--resume")
        # ── Phase 3 annealing args (only forwarded when present in config) ──────
        if getattr(args, "temp_start", 0.0) > 0:
            cmd += [
                "--temp_start", str(args.temp_start),
                "--temp_end", str(getattr(args, "temp_end", args.temperature)),
            ]
        if getattr(args, "hard_weight_start", -1.0) >= 0:
            cmd += [
                "--hard_weight_start", str(args.hard_weight_start),
                "--hard_weight_end", str(
                    getattr(args, "hard_weight_end", getattr(args, "ce_alpha", 0.2))
                ),
            ]

    else:  # unsloth
        cmd = [
            sys.executable, "-m", "distill.training.backends.unsloth",
            "--output_dir", str(output_dir),
            "--epochs", str(args.epochs),
            "--max_samples", str(args.max_samples),
            "--kd_temp", str(args.temperature),
            "--lora_r", str(args.lora_r),
            "--q_bits", str(args.q_bits),
            "--seed", str(trial_seed),
        ]
        _ds = dataset_override or getattr(args, "dataset", None)
        if _ds:
            cmd += ["--dataset", _ds]
        if args.open:
            cmd.append("--open")
        if args.offline:
            cmd.append("--offline")
        if args.watchdog:
            cmd.append("--watchdog")

    return cmd
