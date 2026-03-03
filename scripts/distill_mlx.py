#!/usr/bin/env python3
"""
MLX knowledge distillation backend (Apple-native, 2-5× faster than PyTorch/MPS).

Uses mlx-lm for lazy evaluation + unified memory. Implements forward KL loss
between teacher and student logits. LoRA adapters applied via mlx_lm.tuner.

Same pause.flag protocol and metrics.jsonl format as distill_minillm.py.

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
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--kd_temp", type=float, default=1.0, help="KD temperature")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--q_bits", type=int, default=4, choices=[4, 8],
                   help="Quantization bits for MLX export (4 or 8)")
    p.add_argument("--offline", action="store_true", help="Air-gapped: local cache only")
    p.add_argument("--watchdog", action="store_true", help="Enable pause.flag monitoring")
    p.add_argument("--no_export", action="store_true", help="Skip MLX quantization export")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
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


def _load_dataset(args):
    """Load dataset, respecting offline/air-gapped mode."""
    if args.offline or os.environ.get("HF_DATASETS_OFFLINE") == "1":
        cache_candidates = [
            Path("datasets_cache") / args.dataset.replace("/", "___"),
            Path("scripts/datasets_cache") / args.dataset.replace("/", "___"),
        ]
        for c in cache_candidates:
            if c.exists():
                from datasets import load_from_disk
                LOG.info("Loading dataset from disk: %s", c)
                return load_from_disk(str(c))
        raise FileNotFoundError(
            "Offline mode: dataset not found in cache. Run scripts/cache_datasets.py first."
        )

    from datasets import load_dataset
    LOG.info("Loading dataset: %s", args.dataset)
    return load_dataset(args.dataset)


def _format_prompt(example):
    prompt = example.get("instruction", example.get("prompt", ""))
    if example.get("input"):
        prompt += "\n\nInput: " + example["input"]
    output = example.get("output", example.get("response", ""))
    return prompt + "\n\n### Response:\n" + output


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


def _tokenize_batch(tokenizer, texts, max_length=512):
    """Tokenize a batch of texts → MLX arrays.

    Accepts either an mlx_lm TokenizerWrapper (uses ._tokenizer) or a plain HF tokenizer.
    """
    import mlx.core as mx
    # mlx_lm returns a TokenizerWrapper; unwrap to the underlying HF tokenizer
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    enc = hf_tok(
        texts,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return mx.array(enc["input_ids"]), mx.array(enc["attention_mask"])


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
    ds = _load_dataset(args)
    train_split = ds["train"] if hasattr(ds, "__getitem__") and "train" in ds else ds
    if args.max_samples and len(train_split) > args.max_samples:
        train_split = train_split.select(range(args.max_samples))
    LOG.info("Train samples: %d", len(train_split))

    texts = [_format_prompt(ex) for ex in train_split]

    # ── Optimizer ───────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(learning_rate=args.learning_rate)

    # ── KD loss (forward KL) — defined as a module method for value_and_grad ────
    def kd_loss(model, input_ids, attention_mask, teacher_logits_frozen):
        """Forward KL: KL(teacher || student) per masked token."""
        s_out = model(input_ids)
        s_logits = (s_out if isinstance(s_out, mx.array) else s_out.logits) / args.kd_temp

        t_logits = teacher_logits_frozen / args.kd_temp

        s_log_probs = nn.log_softmax(s_logits, axis=-1)
        t_probs = nn.softmax(t_logits, axis=-1)

        # Forward KL: sum over vocab
        mask = attention_mask[..., None].astype(mx.float32)
        kl = (t_probs * (mx.log(t_probs + 1e-9) - s_log_probs)) * mask
        return kl.sum(axis=-1).mean()

    loss_and_grad = nn.value_and_grad(student_model, kd_loss)

    # ── Training loop ────────────────────────────────────────────────────────────
    n_samples = len(texts)
    batch_size = args.batch_size
    steps_per_epoch = max(1, n_samples // batch_size)
    total_steps = args.epochs * steps_per_epoch

    LOG.info(
        "Starting MLX KD training: epochs=%d  steps/epoch=%d  total_steps=%d",
        args.epochs, steps_per_epoch, total_steps,
    )

    global_step = 0
    t0 = time.time()
    tokenizer = student_tokenizer
    log_history = []  # accumulates entries for trainer_state.json

    for epoch in range(args.epochs):
        indices = list(range(n_samples))
        random.shuffle(indices)

        for batch_start in range(0, n_samples - batch_size + 1, batch_size):
            if args.watchdog and _check_pause_flag(output_dir):
                LOG.info("Saving student weights before exit.")
                student_model.save_weights(str(output_dir / "mlx_student_weights.npz"))
                return

            batch_idx = indices[batch_start: batch_start + batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            input_ids, attention_mask = _tokenize_batch(tokenizer, batch_texts)

            # Get teacher logits (frozen — no gradient flows through teacher)
            t_out = teacher_model(input_ids)
            teacher_logits = (t_out if isinstance(t_out, mx.array) else t_out.logits)
            mx.eval(teacher_logits)  # Materialise before student forward

            loss, grads = loss_and_grad(student_model, input_ids, attention_mask, teacher_logits)
            optimizer.update(student_model, grads)
            mx.eval(student_model.parameters(), optimizer.state, loss)

            global_step += 1
            epoch_frac = epoch + batch_start / n_samples

            if global_step % args.log_steps == 0:
                elapsed = time.time() - t0
                loss_val = float(loss)
                LOG.info(
                    "step=%d  epoch=%.2f  loss=%.4f  %.2f steps/s",
                    global_step, epoch_frac, loss_val,
                    global_step / max(elapsed, 1e-6),
                )
                _write_metric(metrics_path, global_step, epoch_frac, loss=loss_val)
                log_history.append({"step": global_step, "epoch": epoch_frac, "loss": loss_val})
                _write_trainer_state(trainer_state_path, log_history)

            if global_step % args.eval_steps == 0:
                eval_texts = texts[:min(32, len(texts))]
                eval_ids, eval_mask = _tokenize_batch(tokenizer, eval_texts)
                t_eval = teacher_model(eval_ids)
                t_eval_logits = (t_eval if isinstance(t_eval, mx.array) else t_eval.logits)
                mx.eval(t_eval_logits)
                eval_loss, _ = loss_and_grad(student_model, eval_ids, eval_mask, t_eval_logits)
                mx.eval(eval_loss)
                eval_loss_val = float(eval_loss)
                LOG.info("  eval_loss=%.4f", eval_loss_val)
                _write_metric(metrics_path, global_step, epoch_frac, eval_loss=eval_loss_val)
                log_history.append({"step": global_step, "epoch": epoch_frac, "eval_loss": eval_loss_val})
                _write_trainer_state(trainer_state_path, log_history)

            if global_step >= total_steps:
                break
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
