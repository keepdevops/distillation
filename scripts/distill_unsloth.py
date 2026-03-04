#!/usr/bin/env python3
"""
Unsloth-backed knowledge distillation (MLX-optimized LoRA + KD).

Uses Unsloth's FastLanguageModel for the student (optimized LoRA kernels),
and mlx-lm for the frozen teacher. Injects KD loss into trl.SFTTrainer.

Same pause.flag + metrics.jsonl protocol as other backends → dashboard unchanged.

Usage:
  python scripts/distill_unsloth.py --open --max_samples 100 --epochs 1
  python scripts/distill_unsloth.py --teacher Qwen/Qwen2-1.5B-Instruct \\
      --student Qwen/Qwen2-0.5B-Instruct --output_dir ./distilled-unsloth

Requirements (optional):
  pip install "unsloth[mlx]"      # Apple Silicon MLX backend
  pip install "unsloth"           # CPU/CUDA fallback
"""

import argparse
import json
import logging
import os
import sys
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
    p = argparse.ArgumentParser(description="Unsloth knowledge distillation")
    p.add_argument("--teacher", type=str, default="meta-llama/Llama-3.2-8B-Instruct")
    p.add_argument("--student", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--open", action="store_true", help="Use open Qwen2 models (no HF login)")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--output_dir", type=str, default="./distilled-unsloth")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8, tuned for M3 Max)")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--kd_temp", type=float, default=1.0, help="KD temperature")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--q_bits", type=int, default=4, choices=[4, 8],
                   help="Student load quantization (4-bit or 8-bit via Unsloth)")
    p.add_argument("--offline", action="store_true", help="Air-gapped: local cache only")
    p.add_argument("--watchdog", action="store_true", help="Enable pause.flag monitoring")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
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


def _load_dataset_texts(args):
    """Load dataset and return list of formatted text strings."""
    if args.offline or os.environ.get("HF_DATASETS_OFFLINE") == "1":
        cache_candidates = [
            Path("datasets_cache") / args.dataset.replace("/", "___"),
            Path("scripts/datasets_cache") / args.dataset.replace("/", "___"),
        ]
        for c in cache_candidates:
            if c.exists():
                from datasets import load_from_disk
                LOG.info("Loading dataset from disk: %s", c)
                ds = load_from_disk(str(c))
                split = ds["train"] if "train" in ds else ds
                break
        else:
            raise FileNotFoundError(
                "Offline mode: dataset not found in cache. Run scripts/cache_datasets.py first."
            )
    else:
        from datasets import load_dataset
        LOG.info("Downloading dataset: %s", args.dataset)
        ds = load_dataset(args.dataset)
        split = ds["train"]

    if args.max_samples and len(split) > args.max_samples:
        split = split.select(range(args.max_samples))

    texts = []
    for ex in split:
        prompt = ex.get("instruction", ex.get("prompt", ""))
        if ex.get("input"):
            prompt += "\n\nInput: " + ex["input"]
        output = ex.get("output", ex.get("response", ""))
        texts.append(prompt + "\n\n### Response:\n" + output)

    LOG.info("Loaded %d samples.", len(texts))
    return texts


class UnslothKDTrainer:
    """
    Minimal KD trainer wrapping Unsloth student + MLX teacher.
    Implements its own training loop (not SFTTrainer subclass) to avoid
    trl version incompatibilities while keeping the same metrics.jsonl output.
    """

    def __init__(self, student_model, student_tokenizer, teacher_model,
                 teacher_tokenizer, texts, args, output_dir: Path):
        self.student = student_model
        self.student_tok = student_tokenizer
        self.teacher = teacher_model
        self.teacher_tok = teacher_tokenizer
        self.texts = texts
        self.args = args
        self.output_dir = output_dir
        self.metrics_path = output_dir / "metrics.jsonl"

    def _check_pause(self):
        flag = self.output_dir / "pause.flag"
        if not flag.exists():
            return False
        try:
            info = json.load(open(flag))
            reason = info.get("reason", "unknown")
        except (json.JSONDecodeError, OSError):
            reason = "pause.flag"
        LOG.info("pause.flag detected (reason=%s). Stopping.", reason)
        return True

    def _write_metric(self, step, epoch, **kwargs):
        row = {"step": step, "epoch": epoch, **kwargs}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _get_teacher_logits(self, input_ids_np):
        """Get teacher logits via MLX (frozen, no-grad)."""
        import mlx.core as mx
        import mlx.nn as nn
        mx_ids = mx.array(input_ids_np)
        out = self.teacher(mx_ids)
        logits = out if isinstance(out, mx.array) else out.logits
        return logits  # MLX array

    def train(self):
        import torch
        import torch.nn.functional as F

        args = self.args
        tokenizer = self.student_tok
        model = self.student

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        n = len(self.texts)
        batch_size = args.batch_size
        steps_per_epoch = max(1, n // batch_size)
        total_steps = args.epochs * steps_per_epoch

        LOG.info(
            "Unsloth KD training: epochs=%d  steps/epoch=%d  total=%d",
            args.epochs, steps_per_epoch, total_steps,
        )

        global_step = 0
        import random
        import time
        t0 = time.time()

        for epoch in range(args.epochs):
            idx = list(range(n))
            random.shuffle(idx)

            for batch_start in range(0, n, batch_size):
                if args.watchdog and self._check_pause():
                    LOG.info("Saving model before pause exit.")
                    model.save_pretrained(str(self.output_dir))
                    tokenizer.save_pretrained(str(self.output_dir))
                    return

                batch_texts = [self.texts[i] for i in idx[batch_start: batch_start + batch_size]]
                enc = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]

                # Student forward (PyTorch / Unsloth)
                s_out = model(input_ids=input_ids, attention_mask=attention_mask)
                s_logits = s_out.logits / args.kd_temp  # (B, T, V)

                # Teacher forward (MLX)
                t_logits_mx = self._get_teacher_logits(input_ids.numpy())
                # Convert to torch
                import numpy as np
                t_logits_np = np.array(t_logits_mx)
                t_logits = torch.tensor(t_logits_np, dtype=s_logits.dtype,
                                        device=s_logits.device) / args.kd_temp

                # Match vocab size (teacher may differ from student)
                min_vocab = min(s_logits.shape[-1], t_logits.shape[-1])
                s_logits = s_logits[..., :min_vocab]
                t_logits = t_logits[..., :min_vocab]

                # Forward KL loss: KL(t || s) = sum(t * (log t - log s))
                t_probs = F.softmax(t_logits, dim=-1)
                s_log_probs = F.log_softmax(s_logits, dim=-1)
                kl = (t_probs * (torch.log(t_probs + 1e-9) - s_log_probs))

                # Mask padding tokens
                mask = attention_mask.unsqueeze(-1).float()
                loss = (kl * mask).sum(dim=-1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1
                epoch_frac = epoch + batch_start / n

                if global_step % 10 == 0:
                    elapsed = time.time() - t0
                    loss_val = loss.item()
                    LOG.info(
                        "step=%d  epoch=%.2f  loss=%.4f  %.2f s",
                        global_step, epoch_frac, loss_val, elapsed,
                    )
                    self._write_metric(global_step, epoch_frac, loss=loss_val)

                if global_step % args.eval_steps == 0:
                    # Quick eval on first 32 samples
                    with torch.no_grad():
                        eval_texts = self.texts[:min(32, len(self.texts))]
                        enc_eval = tokenizer(
                            eval_texts, return_tensors="pt",
                            padding=True, truncation=True, max_length=512,
                        )
                        e_out = model(
                            input_ids=enc_eval["input_ids"],
                            attention_mask=enc_eval["attention_mask"],
                        )
                        e_logits = e_out.logits / args.kd_temp
                        t_mx = self._get_teacher_logits(enc_eval["input_ids"].numpy())
                        t_np = np.array(t_mx)
                        t_eval = torch.tensor(t_np, dtype=e_logits.dtype) / args.kd_temp
                        mv = min(e_logits.shape[-1], t_eval.shape[-1])
                        kl_eval = (
                            F.softmax(t_eval[..., :mv], dim=-1)
                            * (
                                torch.log(F.softmax(t_eval[..., :mv], dim=-1) + 1e-9)
                                - F.log_softmax(e_logits[..., :mv], dim=-1)
                            )
                        ).sum(-1).mean()
                        eval_loss_val = kl_eval.item()
                    LOG.info("  eval_loss=%.4f", eval_loss_val)
                    self._write_metric(global_step, epoch_frac, eval_loss=eval_loss_val)

                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        LOG.info("Saved Unsloth student: %s", self.output_dir)


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
        args.teacher = OPEN_TEACHER
        args.student = OPEN_STUDENT
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
    texts = _load_dataset_texts(args)

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
    main()
