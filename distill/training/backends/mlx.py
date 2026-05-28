#!/usr/bin/env python3
"""MLX knowledge distillation backend (Apple-native, 2-5x faster than PyTorch/MPS).

Uses mlx-lm for lazy evaluation + unified memory. Implements forward KL loss
between teacher and student logits. LoRA adapters applied via mlx_lm.tuner.

Usage:
  python -m distill.distill_mlx --open --max_samples 100 --epochs 1
  python -m distill.distill_mlx --teacher Qwen/Qwen2-1.5B-Instruct \\
      --student Qwen/Qwen2-0.5B-Instruct --output_dir ./distilled-mlx --q_bits 4
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ...infra.cli_common import add_cache_and_offline
from ...data.pipeline import (
    load_dataset_split,
    format_prompt_full,
    format_multiturn_full,
    pretokenize,
    validate_dataset_schema,
    DATASET_HELP,
)
from .mlx_lora import resolve_mlx_path as _resolve_mlx_path, check_mlx as _check_mlx
from .mlx_trainer_state import write_trainer_state as _write_trainer_state
from .mlx_memory import memory_safe_precomp_bs as _memory_safe_precomp_bs
from .mlx_precompute import precompute_teacher_logits
from .mlx_loss import kd_loss
from .mlx_train_loop import run_training_loop
from ...infra.config import cfg

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    p = argparse.ArgumentParser(description="MLX knowledge distillation")
    p.add_argument("--teacher", type=str, default=cfg.models.default_teacher)
    p.add_argument("--student", type=str, default=cfg.models.default_student)
    p.add_argument("--open", action="store_true", help="Use open Qwen2 models (no HF login)")
    p.add_argument("--dataset", type=str, default=cfg.models.default_dataset, help=DATASET_HELP)
    p.add_argument("--output_dir", type=str, default="./distilled-mlx")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=cfg.training.batch_size_mlx, help="Training batch size (default: 2)")
    p.add_argument("--lora_r", type=int, default=cfg.training.lora_r)
    p.add_argument("--kd_temp", type=float, default=cfg.training.kd_temperature, help="KD temperature")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=cfg.training.learning_rate)
    p.add_argument("--q_bits", type=int, default=4, choices=[4, 8], help="Quantization bits (4 or 8)")
    add_cache_and_offline(p)
    p.add_argument("--watchdog", action="store_true", help="Enable pause.flag monitoring")
    p.add_argument("--no_export", action="store_true", help="Skip MLX quantization export")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--topk_logits", type=int, default=cfg.training.topk_logits,
                   help="Top-K teacher logits kept per token (default: 50, >99%% probability mass)")
    p.add_argument("--grad_acc", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size x grad_acc, default: 4)")
    p.add_argument("--ce_alpha", type=float, default=0.1,
                   help="CE weight in mixed KD+CE loss (0=pure KD, 1=pure CE, default: 0.1)")
    p.add_argument("--resume", action="store_true", help="Resume from last epoch checkpoint")
    p.add_argument("--multi_turn_ratio", type=float, default=0.0,
                   help="Fraction of samples as multi-turn ChatML (0.0=all single-turn, default: 0.0)")
    p.add_argument("--precomp_bs", type=int, default=0,
                   help="Teacher precompute batch size (0=auto: 4x batch_size min 8, default: 0)")
    # Phase 3: temperature / alpha annealing
    p.add_argument("--temp_start", type=float, default=0.0,
                   help="KD temp at step 0; anneal to --temp_end if >0 (default: 0 = fixed)")
    p.add_argument("--temp_end", type=float, default=0.0,
                   help="KD temp at final step (used when temp_start > 0, default: 0)")
    p.add_argument("--hard_weight_start", type=float, default=-1.0,
                   help="CE alpha at step 0; anneal if >=0 (-1 = use --ce_alpha fixed, default: -1)")
    p.add_argument("--hard_weight_end", type=float, default=-1.0,
                   help="CE alpha at final step (used when hard_weight_start >=0, default: -1)")
    return p.parse_args()


def _prepare_dataset(args, teacher_model, teacher_tokenizer):
    """Load dataset, format texts, pre-tokenize, and run teacher precompute.

    Returns (all_input_ids, all_attention_mask, all_teacher_topk_values,
              all_teacher_topk_indices, n_samples, seq_len, K).
    """
    import random as _random
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    LOG.info("Loading dataset: %s", args.dataset)
    train_split = load_dataset_split(args.dataset, args.max_samples, ds_cache, args.offline)
    LOG.info("Train samples: %d", len(train_split))
    validate_dataset_schema(train_split, args.dataset, logger=LOG)

    if args.multi_turn_ratio > 0.0:
        texts = []
        for ex in train_split:
            if _random.random() < args.multi_turn_ratio:
                texts.append(format_multiturn_full(ex, max_turns=4))
            else:
                texts.append(format_prompt_full(ex))
        n_mt = sum(1 for t in texts if "<|im_start|>" in t)
        LOG.info("Multi-turn formatting: %d/%d samples (ratio=%.2f)",
                 n_mt, len(texts), args.multi_turn_ratio)
    else:
        texts = [format_prompt_full(ex) for ex in train_split]

    LOG.info("Pre-tokenizing %d samples...", len(texts))
    all_input_ids, all_attention_mask = pretokenize(teacher_tokenizer, texts)
    n_samples, seq_len = len(texts), all_input_ids.shape[1]
    LOG.info("Pre-tokenization complete.")

    K = args.topk_logits
    _initial_bs = args.precomp_bs if args.precomp_bs > 0 else min(max(args.batch_size * 4, 8), 32)
    vocab_size = (
        teacher_model.model.embed_tokens.weight.shape[0]
        if hasattr(getattr(teacher_model, "model", None), "embed_tokens")
        else len(teacher_tokenizer)
    )
    precomp_bs = _memory_safe_precomp_bs(_initial_bs, seq_len, vocab_size)
    LOG.info("Pre-computing teacher top-%d logits for %d samples (precomp_bs=%d, vocab=%d)...",
             K, n_samples, precomp_bs, vocab_size)
    topk_values, topk_indices = precompute_teacher_logits(
        teacher_model, all_input_ids, n_samples, seq_len, K, precomp_bs, logger=LOG,
    )
    return all_input_ids, all_attention_mask, topk_values, topk_indices, n_samples, seq_len, K


def _setup_student(args, student_path, mlx_load, linear_to_lora_layers):
    """Load student model and apply LoRA. Returns (student_model, student_tokenizer)."""
    LOG.info("Loading student: %s", student_path)
    student_model, student_tokenizer = mlx_load(student_path)
    LOG.info("Applying LoRA (r=%d) to student...", args.lora_r)
    lora_config = {"rank": args.lora_r, "scale": 20.0, "dropout": 0.0}
    num_blocks = len(list(student_model.model.layers)) if hasattr(student_model, "model") else 24
    linear_to_lora_layers(student_model, num_blocks, lora_config)
    student_model.train()
    trainable = list(student_model.trainable_parameters().items())
    LOG.info("LoRA applied: %d trainable parameter tensors.", len(trainable))
    return student_model, student_tokenizer


def main():  # noqa: C901
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    if not _check_mlx():
        sys.exit(1)

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import random as _random
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner import linear_to_lora_layers

    _random.seed(args.seed)
    mx.random.seed(args.seed)

    try:
        _minfo = mx.metal.device_info()
        LOG.info("Metal memory: total=%.0f GB  recommended_max=%.0f GB",
                 _minfo.get("memorySize", 0) / 1e9,
                 _minfo.get("recommendedMaxWorkingSetSize", 0) / 1e9)
    except Exception as e:
        LOG.error("Could not query Metal device info: %s", e)

    if args.open:
        args.teacher, args.student = cfg.models.open_teacher, cfg.models.open_student
        LOG.info("Using open models: teacher=%s  student=%s", args.teacher, args.student)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    trainer_state_path = output_dir / "trainer_state.json"
    _write_trainer_state(trainer_state_path, [])

    teacher_path = _resolve_mlx_path(args.teacher)
    LOG.info("Loading teacher: %s", teacher_path)
    teacher_model, teacher_tokenizer = mlx_load(teacher_path)
    teacher_model.freeze()
    LOG.info("Teacher loaded and frozen.")

    (all_input_ids, all_attention_mask,
     all_teacher_topk_values, all_teacher_topk_indices,
     n_samples, seq_len, K) = _prepare_dataset(args, teacher_model, teacher_tokenizer)

    del teacher_model
    mx.clear_cache()

    student_model, _ = _setup_student(args, _resolve_mlx_path(args.student), mlx_load, linear_to_lora_layers)

    eval_size = min(32, n_samples)
    eval_tensors = (
        mx.array(all_input_ids[:eval_size]),
        mx.array(all_attention_mask[:eval_size]),
        mx.array(all_teacher_topk_values[:eval_size].astype(np.float32)),
        mx.array(all_teacher_topk_indices[:eval_size]),
    )

    _macro = args.batch_size * args.grad_acc
    _steps_per_epoch = max(1, n_samples // _macro)
    _total_steps = args.epochs * _steps_per_epoch
    _warmup_steps = max(1, int(0.03 * _total_steps))
    _cosine_steps = max(1, _total_steps - _warmup_steps)
    lr_schedule = optim.join_schedules(
        [optim.linear_schedule(1e-7, args.learning_rate, _warmup_steps),
         optim.cosine_decay(args.learning_rate, _cosine_steps)],
        [_warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule)
    LOG.info("LR schedule: warmup %d steps -> cosine decay over %d steps", _warmup_steps, _cosine_steps)

    _do_temp_anneal = args.temp_start > 0.0 and args.temp_end > 0.0
    _do_alpha_anneal = args.hard_weight_start >= 0.0 and args.hard_weight_end >= 0.0

    def _anneal(start, end, step, total):
        return start + (end - start) * min(step / max(total, 1), 1.0)

    def _current_temp(step):
        return _anneal(args.temp_start, args.temp_end, step, _total_steps) if _do_temp_anneal else args.kd_temp

    def _current_alpha(step):
        return _anneal(args.hard_weight_start, args.hard_weight_end, step, _total_steps) if _do_alpha_anneal else args.ce_alpha

    if _do_temp_anneal:
        LOG.info("Temperature annealing: %.2f -> %.2f over %d steps", args.temp_start, args.temp_end, _total_steps)
    if _do_alpha_anneal:
        LOG.info("CE-alpha annealing: %.2f -> %.2f over %d steps", args.hard_weight_start, args.hard_weight_end, _total_steps)

    loss_and_grad = nn.value_and_grad(student_model, kd_loss)

    start_epoch, global_step = 0, 0
    checkpoint_path = output_dir / "checkpoint.json"
    if args.resume and checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("global_step", 0)
            weights_file = output_dir / "mlx_student_weights.npz"
            if weights_file.exists():
                student_model.load_weights(str(weights_file))
                LOG.info("Resumed: epoch=%d  global_step=%d", ckpt.get("epoch", 0), global_step)
            else:
                LOG.warning("--resume set but no weights file found; starting fresh.")
                start_epoch, global_step = 0, 0
        except Exception as e:
            LOG.error("Failed to load checkpoint from %s: %s", checkpoint_path, e)
            raise

    LOG.info("Starting MLX KD training: epochs=%d  steps/epoch=%d  total=%d  batch=%d  "
             "grad_acc=%d  effective_batch=%d  topk=%d",
             args.epochs, _steps_per_epoch, _total_steps,
             args.batch_size, args.grad_acc, _macro, K)

    run_training_loop(
        args=args, student_model=student_model, optimizer=optimizer,
        all_input_ids=all_input_ids, all_attention_mask=all_attention_mask,
        all_teacher_topk_values=all_teacher_topk_values,
        all_teacher_topk_indices=all_teacher_topk_indices,
        eval_tensors=eval_tensors, loss_and_grad=loss_and_grad,
        kd_loss_fn=kd_loss, current_temp_fn=_current_temp, current_alpha_fn=_current_alpha,
        metrics_path=metrics_path, trainer_state_path=trainer_state_path,
        output_dir=output_dir, total_steps=_total_steps, steps_per_epoch=_steps_per_epoch,
        start_epoch=start_epoch, start_global_step=global_step,
    )

    weights_path = output_dir / "mlx_student_weights.npz"
    try:
        student_model.save_weights(str(weights_path))
        LOG.info("Saved MLX student weights: %s", weights_path)
    except Exception as e:
        LOG.error("Failed to save final student weights to %s: %s", weights_path, e)
        raise

    with open(output_dir / "distill_config.json", "w") as f:
        json.dump({"teacher": args.teacher, "student": args.student, "lora_r": args.lora_r,
                   "kd_temp": args.kd_temp, "epochs": args.epochs,
                   "max_samples": args.max_samples, "backend": "mlx"}, f, indent=2)

    if not args.no_export:
        LOG.info("Quantizing with mlx_lm.convert (q_bits=%d)...", args.q_bits)
        try:
            import shutil
            from mlx_lm import convert as mlx_convert
            quant_dir = output_dir / f"mlx_q{args.q_bits}"
            if quant_dir.exists():
                LOG.info("Removing existing quantized dir: %s", quant_dir)
                shutil.rmtree(quant_dir)
            mlx_convert(args.student, quantize=True, q_bits=args.q_bits, mlx_path=str(quant_dir))
            LOG.info("Quantized model saved: %s", quant_dir)
        except Exception as e:
            LOG.error("MLX quantization failed: %s", e)
            raise

    LOG.info("MLX distillation complete. Output: %s", output_dir)


if __name__ == "__main__":
    subprocess.Popen(['caffeinate', '-i', 'sleep', '3600'])
    main()
