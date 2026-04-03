#!/usr/bin/env python3
"""
Unsloth-backed knowledge distillation (MLX-optimized LoRA + KD).

Uses Unsloth's FastLanguageModel for the student (optimized LoRA kernels),
and mlx-lm for the frozen teacher. Injects KD loss into trl.SFTTrainer.

Same pause.flag + metrics.jsonl protocol as other backends → dashboard unchanged.

Usage:
  python -m distill.distill_unsloth --open --max_samples 100 --epochs 1
  python -m distill.distill_unsloth --teacher Qwen/Qwen2-1.5B-Instruct \\
      --student Qwen/Qwen2-0.5B-Instruct --output_dir ./distilled-unsloth

Requirements (optional):
  pip install "unsloth[mlx]"      # Apple Silicon MLX backend
  pip install "unsloth"           # CPU/CUDA fallback
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from ...data.pipeline import load_dataset_split, format_prompt_full, validate_dataset_schema, DATASET_HELP
from ...infra.config import cfg
from .unsloth_trainer import UnslothKDTrainer  # noqa: F401 — re-exported for callers

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    p = argparse.ArgumentParser(description="Unsloth knowledge distillation")
    p.add_argument("--teacher", type=str, default=cfg.models.default_teacher)
    p.add_argument("--student", type=str, default=cfg.models.default_student)
    p.add_argument("--open", action="store_true", help="Use open Qwen2 models (no HF login)")
    p.add_argument("--dataset", type=str, default=cfg.models.default_dataset, help=DATASET_HELP)
    p.add_argument("--output_dir", type=str, default="./distilled-unsloth")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=cfg.training.batch_size, help="Batch size (default: 8, tuned for M3 Max)")
    p.add_argument("--lora_r", type=int, default=cfg.training.lora_r)
    p.add_argument("--kd_temp", type=float, default=cfg.training.kd_temperature, help="KD temperature")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=cfg.training.learning_rate)
    p.add_argument("--q_bits", type=int, default=4, choices=[4, 8],
                   help="Student load quantization (4-bit or 8-bit via Unsloth)")
    p.add_argument("--offline", action="store_true", help="Air-gapped: local cache only")
    p.add_argument("--watchdog", action="store_true", help="Enable pause.flag monitoring")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    p.add_argument("--topk_logits", type=int, default=cfg.training.topk_logits,
                   help="Top-K teacher logits for distillation (default: 50). ~300 MB vs 311 GB full vocab.")
    p.add_argument("--grad_acc", type=int, default=4,
                   help="Gradient accumulation steps (default: 4, effective batch = batch_size × grad_acc)")
    p.add_argument("--ce_alpha", type=float, default=0.1,
                   help="CE loss weight mixed with KD loss (default: 0.1). 0=pure KD, 1=pure CE.")
    return p.parse_args()


def _check_imports():
    """Check required packages. Returns (unsloth_ok, mlx_ok)."""
    unsloth_ok = False
    mlx_ok = False

    try:
        import unsloth  # noqa: F401
        unsloth_ok = True
    except ImportError:
        pass

    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        mlx_ok = True
    except ImportError:
        pass

    return unsloth_ok, mlx_ok


def main():
    args = parse_args()

    unsloth_ok, mlx_ok = _check_imports()

    if not unsloth_ok:
        LOG.error(
            "unsloth is not installed.\n\n"
            "Install instructions:\n"
            "  Apple Silicon (MLX backend):\n"
            "    pip install 'unsloth[mlx]'\n\n"
            "  CPU / CUDA fallback:\n"
            "    pip install unsloth\n\n"
            "  Air-gapped: download wheel from https://github.com/unslothai/unsloth/releases\n"
            "  then: pip install unsloth-<version>.whl\n\n"
            "Tip: Use --backend pytorch or --backend mlx if Unsloth is unavailable."
        )
        sys.exit(1)

    if not mlx_ok:
        LOG.error(
            "mlx and mlx-lm are required for the teacher model.\n"
            "Install with: pip install mlx mlx-lm"
        )
        sys.exit(1)

    if args.open:
        args.teacher = cfg.models.open_teacher
        args.student = cfg.models.open_student
        LOG.info("Using open models: teacher=%s  student=%s", args.teacher, args.student)

    import random as _random
    _random.seed(args.seed)

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load teacher via MLX (frozen)
    LOG.info("Loading teacher (MLX): %s", args.teacher)
    from mlx_lm import load as mlx_load
    teacher_model, teacher_tokenizer = mlx_load(args.teacher)
    teacher_model.freeze()
    LOG.info("Teacher loaded and frozen.")

    # Load student via Unsloth
    LOG.info("Loading student (Unsloth, %d-bit): %s", args.q_bits, args.student)
    from unsloth import FastLanguageModel

    load_in_4bit = args.q_bits == 4
    load_in_8bit = args.q_bits == 8

    student_model, student_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.student,
        max_seq_length=512,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    student_model = FastLanguageModel.get_peft_model(
        student_model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    LOG.info("Student loaded with Unsloth LoRA.")

    # Dataset
    ds_cache = os.environ.get("HF_DATASETS_CACHE")
    LOG.info("Loading dataset: %s", args.dataset)
    train_split = load_dataset_split(args.dataset, args.max_samples, ds_cache, args.offline)
    validate_dataset_schema(train_split, args.dataset, logger=LOG)
    texts = [format_prompt_full(ex) for ex in train_split]
    LOG.info("Loaded %d samples.", len(texts))

    # Train
    trainer = UnslothKDTrainer(
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        texts=texts,
        args=args,
        output_dir=output_dir,
    )
    trainer.train()

    # Save config
    config = {
        "teacher": args.teacher,
        "student": args.student,
        "lora_r": args.lora_r,
        "kd_temp": args.kd_temp,
        "epochs": args.epochs,
        "max_samples": args.max_samples,
        "backend": "unsloth",
    }
    with open(output_dir / "distill_config.json", "w") as f:
        json.dump(config, f, indent=2)

    LOG.info("Unsloth distillation complete. Output: %s", output_dir)


if __name__ == "__main__":
    subprocess.Popen(['caffeinate', '-i', 'sleep', '3600'])
    main()
