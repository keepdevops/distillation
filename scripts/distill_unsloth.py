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

import numpy as np

from data_pipeline import load_dataset_split, format_prompt_full, pretokenize, validate_dataset_schema, DATASET_HELP

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
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help=DATASET_HELP)
    p.add_argument("--output_dir", type=str, default="./distilled-unsloth")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8, tuned for M3 Max)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--kd_temp", type=float, default=1.0, help="KD temperature")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--q_bits", type=int, default=4, choices=[4, 8],
                   help="Student load quantization (4-bit or 8-bit via Unsloth)")
    p.add_argument("--offline", action="store_true", help="Air-gapped: local cache only")
    p.add_argument("--watchdog", action="store_true", help="Enable pause.flag monitoring")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    p.add_argument("--topk_logits", type=int, default=50,
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


class UnslothKDTrainer:
    """
    Minimal KD trainer wrapping Unsloth student + MLX teacher.
    Implements its own training loop to avoid trl version incompatibilities
    while keeping the same metrics.jsonl output as other backends.
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
            with open(flag) as f:
                info = json.load(f)
            reason = info.get("reason", "unknown")
        except (json.JSONDecodeError, OSError):
            reason = "pause.flag"
        LOG.info("pause.flag detected (reason=%s). Stopping.", reason)
        return True

    def _write_metric(self, step, epoch, **kwargs):
        row = {"step": step, "epoch": epoch, **kwargs}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _precompute_teacher_topk(self, all_input_ids_np, K):
        """Pre-compute teacher top-K logits for all samples once.

        Teacher is frozen and dataset is fixed — no need to repeat this.
        Returns numpy float16 values + int32 indices: ~300 MB for K=50.
        """
        import mlx.core as mx

        n_samples, seq_len = all_input_ids_np.shape
        topk_values = np.zeros((n_samples, seq_len, K), dtype=np.float16)
        topk_indices = np.zeros((n_samples, seq_len, K), dtype=np.int32)

        LOG.info("Pre-computing teacher top-%d logits for %d samples...", K, n_samples)
        for start in range(0, n_samples, self.args.batch_size):
            end = min(start + self.args.batch_size, n_samples)
            mx_ids = mx.array(all_input_ids_np[start:end])
            out = self.teacher(mx_ids)
            t_logits = out if isinstance(out, mx.array) else out.logits
            mx.eval(t_logits)
            topk_idx = mx.argsort(-t_logits, axis=-1)[..., :K]
            topk_val = mx.take_along_axis(t_logits, topk_idx, axis=-1)
            mx.eval(topk_idx, topk_val)
            topk_values[start:end] = np.array(topk_val.astype(mx.float32)).astype(np.float16)
            topk_indices[start:end] = np.array(topk_idx.astype(mx.int32))
            if end % max(self.args.batch_size * 10, 1) == 0 or end == n_samples:
                LOG.info("  Teacher logits: %d/%d samples", end, n_samples)

        mb = (topk_values.nbytes + topk_indices.nbytes) / 1e6
        LOG.info("Teacher top-%d logits cached (%.0f MB).", K, mb)
        return topk_values, topk_indices

    def train(self):
        import random
        import time
        import torch
        import torch.nn.functional as F

        args = self.args
        tokenizer = self.student_tok
        model = self.student
        K = args.topk_logits
        grad_acc = args.grad_acc
        ce_alpha = args.ce_alpha

        # ── Pre-tokenize dataset once ────────────────────────────────────────────
        LOG.info("Pre-tokenizing %d samples...", len(self.texts))
        all_input_ids_np, all_attention_mask_np = pretokenize(tokenizer, self.texts)
        n_samples = len(self.texts)
        LOG.info("Pre-tokenization complete.")

        # ── Pre-compute teacher top-K logits ─────────────────────────────────────
        topk_values_np, topk_indices_np = self._precompute_teacher_topk(all_input_ids_np, K)

        # ── Eval setup (pre-computed, no teacher forward at eval time) ────────────
        eval_size = min(32, n_samples)
        eval_ids  = torch.tensor(all_input_ids_np[:eval_size], dtype=torch.long)
        eval_mask = torch.tensor(all_attention_mask_np[:eval_size], dtype=torch.long)
        eval_topk_v = torch.tensor(topk_values_np[:eval_size].astype(np.float32))
        eval_topk_i = torch.tensor(topk_indices_np[:eval_size].astype(np.int64))

        # ── Optimizer: linear warmup + cosine decay ───────────────────────────────
        macro_batch = args.batch_size * grad_acc
        steps_per_epoch = max(1, n_samples // macro_batch)
        total_steps = args.epochs * steps_per_epoch
        warmup_steps = max(1, int(0.03 * total_steps))

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, total_steps - warmup_steps)),
            ],
            milestones=[warmup_steps],
        )

        LOG.info(
            "Unsloth KD training: epochs=%d  steps/epoch=%d  total=%d  "
            "batch=%d  grad_acc=%d  effective_batch=%d  topk=%d",
            args.epochs, steps_per_epoch, total_steps,
            args.batch_size, grad_acc, macro_batch, K,
        )

        global_step = 0
        t0 = time.time()

        for epoch in range(args.epochs):
            idx = list(range(n_samples))
            random.shuffle(idx)

            for step_start in range(0, n_samples - macro_batch + 1, macro_batch):
                if args.watchdog and self._check_pause():
                    LOG.info("Saving model before pause exit.")
                    model.save_pretrained(str(self.output_dir))
                    tokenizer.save_pretrained(str(self.output_dir))
                    return

                optimizer.zero_grad()
                accum_loss = 0.0

                for acc in range(grad_acc):
                    mini_start = step_start + acc * args.batch_size
                    mini_idx = idx[mini_start: mini_start + args.batch_size]

                    input_ids      = torch.tensor(all_input_ids_np[mini_idx],      dtype=torch.long)
                    attention_mask = torch.tensor(all_attention_mask_np[mini_idx], dtype=torch.long)
                    t_topk_v       = torch.tensor(topk_values_np[mini_idx].astype(np.float32))
                    t_topk_i       = torch.tensor(topk_indices_np[mini_idx].astype(np.int64))

                    s_out    = model(input_ids=input_ids, attention_mask=attention_mask)
                    s_logits = s_out.logits                                              # (B, T, V)

                    # KD loss: forward KL over top-K teacher positions
                    s_log_probs      = F.log_softmax(s_logits / args.kd_temp, dim=-1)   # (B, T, V)
                    s_log_probs_topk = s_log_probs.gather(-1, t_topk_i)                 # (B, T, K)
                    t_probs          = F.softmax(t_topk_v / args.kd_temp, dim=-1)       # (B, T, K)
                    pad_mask         = attention_mask.unsqueeze(-1).float()
                    kl  = (t_probs * (torch.log(t_probs + 1e-9) - s_log_probs_topk)) * pad_mask
                    kd  = kl.sum(dim=-1).mean()

                    # CE loss: next-token prediction (stabilises early training)
                    if ce_alpha > 0.0:
                        ce_mask = attention_mask[:, 1:].float()
                        ce_nll  = F.cross_entropy(
                            s_logits[:, :-1].reshape(-1, s_logits.size(-1)),
                            input_ids[:, 1:].reshape(-1),
                            reduction="none",
                        ).reshape(input_ids[:, 1:].shape)
                        ce   = (ce_nll * ce_mask).sum() / ce_mask.sum().clamp(min=1)
                        loss = ce_alpha * ce + (1.0 - ce_alpha) * kd
                    else:
                        loss = kd

                    (loss / grad_acc).backward()
                    accum_loss += loss.item() / grad_acc

                optimizer.step()
                scheduler.step()
                global_step += 1
                epoch_frac = epoch + step_start / n_samples

                if global_step % 10 == 0:
                    elapsed = time.time() - t0
                    LOG.info(
                        "step=%d  epoch=%.2f  loss=%.4f  %.2f steps/s",
                        global_step, epoch_frac, accum_loss,
                        global_step / max(elapsed, 1e-6),
                    )
                    self._write_metric(global_step, epoch_frac, loss=accum_loss)

                if global_step % args.eval_steps == 0:
                    with torch.no_grad():
                        e_out    = model(input_ids=eval_ids, attention_mask=eval_mask)
                        e_logits = e_out.logits
                        e_log_p  = F.log_softmax(e_logits / args.kd_temp, dim=-1)
                        e_log_p_topk = e_log_p.gather(-1, eval_topk_i)
                        et_probs     = F.softmax(eval_topk_v / args.kd_temp, dim=-1)
                        emask        = eval_mask.unsqueeze(-1).float()
                        kl_eval      = (et_probs * (torch.log(et_probs + 1e-9) - e_log_p_topk)) * emask
                        eval_loss_val = kl_eval.sum(dim=-1).mean().item()
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
    main()
