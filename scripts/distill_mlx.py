#!/usr/bin/env python3
"""
MLX knowledge distillation backend (Apple-native, 2-5× faster than PyTorch/MPS).

Uses mlx-lm for lazy evaluation + unified memory. Implements forward KL loss
between teacher and student logits. LoRA adapters applied via mlx_lm.tuner.

Same pause.flag protocol and metrics.jsonl format as distill_minillm.py.

Note: MLX has its own optimized kernels and does NOT use Flash Attention or
torch.compile() (PyTorch-only features). MLX is already optimized for Apple
Silicon with unified memory and lazy evaluation.

Usage:
  python scripts/distill_mlx.py --open --max_samples 100 --epochs 1
  python scripts/distill_mlx.py --teacher Qwen/Qwen2-1.5B-Instruct \\
      --student Qwen/Qwen2-0.5B-Instruct --output_dir ./distilled-mlx \\
      --q_bits 4 --epochs 2
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from data_pipeline import load_dataset_split, format_prompt_full, pretokenize

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

OPEN_TEACHER = "Qwen/Qwen2-1.5B-Instruct"
OPEN_STUDENT = "Qwen/Qwen2-0.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description="MLX knowledge distillation")
    p.add_argument("--teacher", type=str, default="meta-llama/Llama-3.2-8B-Instruct")
    p.add_argument("--student", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--open", action="store_true", help="Use open Qwen2 models (no HF login)")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--output_dir", type=str, default="./distilled-mlx")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8, tuned for M3 Max)")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--kd_temp", type=float, default=1.0, help="KD temperature")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--q_bits", type=int, default=4, choices=[4, 8],
                   help="Quantization bits for MLX export (4 or 8)")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true", help="Air-gapped: local cache only")
    p.add_argument("--watchdog", action="store_true", help="Enable pause.flag monitoring")
    p.add_argument("--no_export", action="store_true", help="Skip MLX quantization export")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    p.add_argument("--topk_logits", type=int, default=50,
                   help="Top-K teacher logits to keep for distillation (default: 50). "
                        "Captures >99%% of teacher probability mass at ~300 MB vs 311 GB full vocab.")
    p.add_argument("--grad_acc", type=int, default=4,
                   help="Gradient accumulation steps (default: 4, effective batch = batch_size × grad_acc)")
    p.add_argument("--ce_alpha", type=float, default=0.1,
                   help="Weight of cross-entropy loss mixed with KD loss (default: 0.1). "
                        "0 = pure KD, 1 = pure CE. Stabilises training and improves convergence.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the last saved epoch checkpoint in output_dir.")
    return p.parse_args()


def _check_mlx():
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        LOG.error(
            "mlx and mlx-lm are required. Install with:\n"
            "  pip install mlx mlx-lm\n"
            "Note: MLX is Apple-only (M1/M2/M3 Silicon)."
        )
        return False


def _check_pause_flag(output_dir: Path) -> bool:
    """Return True if pause.flag exists (same protocol as PauseFlagCallback)."""
    flag_path = output_dir / "pause.flag"
    if not flag_path.exists():
        return False
    try:
        with open(flag_path) as f:
            info = json.load(f)
        reason = info.get("reason", "unknown")
    except (json.JSONDecodeError, OSError):
        reason = "pause.flag"
    LOG.info("pause.flag detected (reason=%s). Stopping.", reason)
    return True


def _write_metric(metrics_path: Path, step: int, epoch: float, **kwargs):
    """Append a metrics row in the same format as MetricsCallback."""
    row = {"step": step, "epoch": epoch, **kwargs}
    with open(metrics_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _write_trainer_state(state_path: Path, log_history: list):
    """Write trainer_state.json in HuggingFace Trainer format.

    Keeps training_watchdog.py compatible with MLX runs — watchdog reads
    log_history from this file for plateau detection.
    """
    state = {"log_history": log_history}
    tmp = state_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    tmp.replace(state_path)  # atomic replace


def _add_grads(a, b):
    """Recursively add two gradient trees (nested dicts/lists of mx.arrays)."""
    import mlx.core as mx
    if isinstance(a, mx.array):
        return a + b
    if isinstance(a, dict):
        return {k: _add_grads(a[k], b[k]) for k in a}
    if isinstance(a, list):
        return [_add_grads(x, y) for x, y in zip(a, b)]
    return a


def _scale_grads(g, scale):
    """Recursively scale a gradient tree by a scalar."""
    import mlx.core as mx
    if isinstance(g, mx.array):
        return g * scale
    if isinstance(g, dict):
        return {k: _scale_grads(v, scale) for k, v in g.items()}
    if isinstance(g, list):
        return [_scale_grads(x, scale) for x in g]
    return g


def main():
    args = parse_args()

    if not _check_mlx():
        sys.exit(1)

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner import linear_to_lora_layers

    import random as _random
    _random.seed(args.seed)
    mx.random.seed(args.seed)

    if args.open:
        args.teacher = OPEN_TEACHER
        args.student = OPEN_STUDENT
        LOG.info("Using open models: teacher=%s  student=%s", args.teacher, args.student)

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    trainer_state_path = output_dir / "trainer_state.json"
    # Initialise trainer_state.json so training_watchdog.py can monitor this run
    _write_trainer_state(trainer_state_path, [])

    # ── Load models ─────────────────────────────────────────────────────────────
    LOG.info("Loading teacher: %s", args.teacher)
    teacher_model, teacher_tokenizer = mlx_load(args.teacher)
    teacher_model.freeze()
    LOG.info("Teacher loaded and frozen.")

    LOG.info("Loading student: %s", args.student)
    student_model, student_tokenizer = mlx_load(args.student)

    # Apply LoRA to student via mlx_lm helper (operates on last N transformer blocks)
    LOG.info("Applying LoRA (r=%d) to student...", args.lora_r)
    lora_config = {"rank": args.lora_r, "scale": 20.0, "dropout": 0.0}
    # Apply to all transformer blocks (num_layers=-1 means all)
    num_blocks = len(list(student_model.model.layers)) if hasattr(student_model, "model") else 24
    linear_to_lora_layers(student_model, num_blocks, lora_config)
    student_model.train()
    trainable = [(k, v) for k, v in student_model.trainable_parameters().items()]
    LOG.info("LoRA applied: %d trainable parameter tensors.", len(trainable))

    # ── Dataset ─────────────────────────────────────────────────────────────────
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    LOG.info("Loading dataset: %s", args.dataset)
    train_split = load_dataset_split(args.dataset, args.max_samples, ds_cache, args.offline)
    LOG.info("Train samples: %d", len(train_split))

    texts = [format_prompt_full(ex) for ex in train_split]

    # ── Pre-tokenize entire dataset once ────────────────────────────────────────
    LOG.info("Pre-tokenizing %d samples...", len(texts))
    all_input_ids, all_attention_mask = pretokenize(student_tokenizer, texts)
    n_samples = len(texts)
    seq_len = all_input_ids.shape[1]
    LOG.info("Pre-tokenization complete.")

    # ── Pre-compute teacher top-K logits for ALL samples ─────────────────────────
    # Teacher is frozen + dataset is fixed → compute once, reuse every epoch.
    # Storing top-K=50 indices+values uses ~300 MB vs 311 GB for the full vocab.
    K = args.topk_logits
    all_teacher_topk_values = np.zeros((n_samples, seq_len, K), dtype=np.float16)
    all_teacher_topk_indices = np.zeros((n_samples, seq_len, K), dtype=np.int32)

    LOG.info("Pre-computing teacher top-%d logits for %d samples...", K, n_samples)
    for precomp_start in range(0, n_samples, args.batch_size):
        precomp_end = min(precomp_start + args.batch_size, n_samples)
        batch_ids = mx.array(all_input_ids[precomp_start:precomp_end])
        t_out = teacher_model(batch_ids)
        t_logits = t_out if isinstance(t_out, mx.array) else t_out.logits  # (B, T, V)
        mx.eval(t_logits)
        topk_idx = mx.argsort(-t_logits, axis=-1)[..., :K]           # (B, T, K)
        topk_val = mx.take_along_axis(t_logits, topk_idx, axis=-1)   # (B, T, K)
        mx.eval(topk_idx, topk_val)
        all_teacher_topk_values[precomp_start:precomp_end] = np.array(topk_val.astype(mx.float32)).astype(np.float16)
        all_teacher_topk_indices[precomp_start:precomp_end] = np.array(topk_idx.astype(mx.int32))
        if precomp_end % max(args.batch_size * 10, 1) == 0 or precomp_end == n_samples:
            LOG.info("  Teacher logits: %d/%d samples", precomp_end, n_samples)

    cache_mb = (all_teacher_topk_values.nbytes + all_teacher_topk_indices.nbytes) / 1e6
    LOG.info("Teacher top-%d logits cached (%.0f MB).", K, cache_mb)

    # Free teacher from memory — logits are fully cached, teacher not needed during training
    del teacher_model
    mx.clear_cache()

    # Eval setup — first 32 samples, using pre-computed top-K (no teacher forward at eval time)
    eval_size = min(32, n_samples)
    eval_ids = mx.array(all_input_ids[:eval_size])
    eval_mask = mx.array(all_attention_mask[:eval_size])
    eval_topk_values = mx.array(all_teacher_topk_values[:eval_size].astype(np.float32))
    eval_topk_indices = mx.array(all_teacher_topk_indices[:eval_size])

    # ── Optimizer with linear warmup + cosine decay ──────────────────────────────
    # Compute total_steps here so the LR schedule can be sized correctly.
    _macro = args.batch_size * args.grad_acc
    _steps_per_epoch = max(1, n_samples // _macro)
    _total_steps = args.epochs * _steps_per_epoch
    _warmup_steps = max(1, int(0.03 * _total_steps))
    _cosine_steps = max(1, _total_steps - _warmup_steps)
    warmup_sched = optim.linear_schedule(1e-7, args.learning_rate, _warmup_steps)
    cosine_sched = optim.cosine_decay(args.learning_rate, _cosine_steps)
    lr_schedule = optim.join_schedules([warmup_sched, cosine_sched], [_warmup_steps])
    optimizer = optim.AdamW(learning_rate=lr_schedule)
    LOG.info("LR schedule: warmup %d steps → cosine decay over %d steps", _warmup_steps, _cosine_steps)

    # ── KD loss: forward KL (top-K sparse) + optional CE ────────────────────────
    def kd_loss(model, input_ids, attention_mask, t_topk_values, t_topk_indices):
        """Mixed loss: ce_alpha * CE + (1 - ce_alpha) * forward-KL.

        t_topk_values:  (B, T, K) float32 — raw teacher logits at top-K vocab positions
        t_topk_indices: (B, T, K) int32   — vocab indices of those K positions

        CE term: next-token prediction loss (student predicts input_ids[t+1] at step t).
        KD term: forward KL between teacher's truncated top-K distribution and student.
        Mixing with ce_alpha ∈ [0, 1] (default 0.1) prevents mode collapse and
        stabilises early training.
        """
        s_out = model(input_ids)
        s_logits = s_out if isinstance(s_out, mx.array) else s_out.logits       # (B, T, V)

        # ── KD term ──────────────────────────────────────────────────────────────
        s_log_probs = nn.log_softmax(s_logits / args.kd_temp, axis=-1)          # (B, T, V)
        s_log_probs_topk = mx.take_along_axis(s_log_probs, t_topk_indices, axis=-1)  # (B, T, K)
        t_probs = nn.softmax(t_topk_values / args.kd_temp, axis=-1)             # (B, T, K)
        mask = attention_mask[..., None].astype(mx.float32)                      # (B, T, 1)
        kl = (t_probs * (mx.log(t_probs + 1e-9) - s_log_probs_topk)) * mask
        kd = kl.sum(axis=-1).mean()

        if args.ce_alpha == 0.0:
            return kd

        # ── CE term: next-token prediction ───────────────────────────────────────
        ce_logits  = s_logits[:, :-1]                                            # (B, T-1, V)
        ce_targets = input_ids[:, 1:][..., None]                                 # (B, T-1, 1)
        ce_mask    = attention_mask[:, 1:].astype(mx.float32)                    # (B, T-1)
        ce_log_p   = nn.log_softmax(ce_logits, axis=-1)                          # (B, T-1, V)
        ce_nll     = -mx.take_along_axis(ce_log_p, ce_targets, axis=-1).squeeze(-1)  # (B, T-1)
        ce = (ce_nll * ce_mask).sum() / mx.maximum(ce_mask.sum(), 1.0)

        return args.ce_alpha * ce + (1.0 - args.ce_alpha) * kd

    loss_and_grad = nn.value_and_grad(student_model, kd_loss)

    # ── Checkpoint resume ────────────────────────────────────────────────────────
    checkpoint_path = output_dir / "checkpoint.json"
    start_epoch = 0
    global_step = 0
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        weights_file = output_dir / "mlx_student_weights.npz"
        if weights_file.exists():
            student_model.load_weights(str(weights_file))
            LOG.info("Resumed from checkpoint: epoch %d  global_step %d",
                     ckpt.get("epoch", 0), global_step)
        else:
            LOG.warning("--resume set but no weights file found; starting fresh.")
            start_epoch = 0
            global_step = 0

    # ── Training loop ────────────────────────────────────────────────────────────
    batch_size = args.batch_size
    grad_acc = args.grad_acc
    macro_batch = batch_size * grad_acc           # samples consumed per optimizer step
    steps_per_epoch = max(1, n_samples // macro_batch)
    total_steps = args.epochs * steps_per_epoch

    LOG.info(
        "Starting MLX KD training: epochs=%d  steps/epoch=%d  total_steps=%d  "
        "batch=%d  grad_acc=%d  effective_batch=%d  topk=%d",
        args.epochs, steps_per_epoch, total_steps,
        batch_size, grad_acc, macro_batch, K,
    )

    t0 = time.time()
    log_history = []

    for epoch in range(start_epoch, args.epochs):
        indices = list(range(n_samples))
        random.shuffle(indices)

        for step_start in range(0, n_samples - macro_batch + 1, macro_batch):
            if args.watchdog and _check_pause_flag(output_dir):
                LOG.info("Saving student weights before exit.")
                student_model.save_weights(str(output_dir / "mlx_student_weights.npz"))
                return

            # Accumulate gradients over grad_acc micro-batches
            accum_grads = None
            accum_loss = 0.0
            for acc in range(grad_acc):
                mini_start = step_start + acc * batch_size
                mini_idx = indices[mini_start: mini_start + batch_size]
                input_ids      = mx.array(all_input_ids[mini_idx])
                attention_mask = mx.array(all_attention_mask[mini_idx])
                t_topk_v       = mx.array(all_teacher_topk_values[mini_idx].astype(np.float32))
                t_topk_i       = mx.array(all_teacher_topk_indices[mini_idx])

                loss_val, grads = loss_and_grad(
                    student_model, input_ids, attention_mask, t_topk_v, t_topk_i
                )
                mx.eval(loss_val, grads)
                accum_loss += float(loss_val) / grad_acc
                accum_grads = grads if accum_grads is None else _add_grads(accum_grads, grads)
                mx.eval(accum_grads)

            accum_grads = _scale_grads(accum_grads, 1.0 / grad_acc)
            optimizer.update(student_model, accum_grads)
            mx.eval(student_model.parameters(), optimizer.state)

            global_step += 1
            epoch_frac = epoch + step_start / n_samples

            if global_step % args.log_steps == 0:
                elapsed = time.time() - t0
                LOG.info(
                    "step=%d  epoch=%.2f  loss=%.4f  %.2f steps/s",
                    global_step, epoch_frac, accum_loss,
                    global_step / max(elapsed, 1e-6),
                )
                _write_metric(metrics_path, global_step, epoch_frac, loss=accum_loss)
                log_history.append({"step": global_step, "epoch": epoch_frac, "loss": accum_loss})
                _write_trainer_state(trainer_state_path, log_history)

            if global_step % args.eval_steps == 0:
                eval_loss, _ = loss_and_grad(
                    student_model, eval_ids, eval_mask, eval_topk_values, eval_topk_indices
                )
                mx.eval(eval_loss)
                eval_loss_val = float(eval_loss)
                LOG.info("  eval_loss=%.4f", eval_loss_val)
                _write_metric(metrics_path, global_step, epoch_frac, eval_loss=eval_loss_val)
                log_history.append({"step": global_step, "epoch": epoch_frac, "eval_loss": eval_loss_val})
                _write_trainer_state(trainer_state_path, log_history)

            if global_step >= total_steps:
                break

        # Save epoch checkpoint (enables --resume on crash/interrupt)
        student_model.save_weights(str(output_dir / "mlx_student_weights.npz"))
        with open(checkpoint_path, "w") as f:
            json.dump({"epoch": epoch, "global_step": global_step}, f)
        LOG.info("Checkpoint saved: epoch=%d  global_step=%d", epoch, global_step)

        if global_step >= total_steps:
            break

    # ── Save weights ─────────────────────────────────────────────────────────────
    weights_path = output_dir / "mlx_student_weights.npz"
    student_model.save_weights(str(weights_path))
    LOG.info("Saved MLX student weights: %s", weights_path)

    config = {
        "teacher": args.teacher,
        "student": args.student,
        "lora_r": args.lora_r,
        "kd_temp": args.kd_temp,
        "epochs": args.epochs,
        "max_samples": args.max_samples,
        "backend": "mlx",
    }
    with open(output_dir / "distill_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Optional: MLX quantization ──────────────────────────────────────────────
    if not args.no_export:
        LOG.info("Quantizing with mlx_lm.convert (q_bits=%d)...", args.q_bits)
        try:
            import shutil
            from mlx_lm import convert as mlx_convert
            quant_dir = output_dir / f"mlx_q{args.q_bits}"
            if quant_dir.exists():
                LOG.info("Removing existing quantized dir: %s", quant_dir)
                shutil.rmtree(quant_dir)
            mlx_convert(
                args.student,
                quantize=True,
                q_bits=args.q_bits,
                mlx_path=str(quant_dir),
            )
            LOG.info("Quantized model saved: %s", quant_dir)
        except Exception as e:
            LOG.warning("mlx_lm.convert failed (non-fatal): %s", e)

    LOG.info("MLX distillation complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
