"""UnslothKDTrainer: Unsloth student + MLX teacher knowledge distillation."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ...infra.train_utils import check_pause_flag, write_metric
from ...data.pipeline import pretokenize

LOG = logging.getLogger(__name__)


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
        return check_pause_flag(self.output_dir)

    def _write_metric(self, step, epoch, **kwargs):
        write_metric(self.metrics_path, step, epoch, **kwargs)

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

        LOG.info("Pre-tokenizing %d samples...", len(self.texts))
        all_input_ids_np, all_attention_mask_np = pretokenize(tokenizer, self.texts)
        n_samples = len(self.texts)
        LOG.info("Pre-tokenization complete.")

        topk_values_np, topk_indices_np = self._precompute_teacher_topk(all_input_ids_np, K)

        eval_size = min(32, n_samples)
        eval_ids = torch.tensor(all_input_ids_np[:eval_size], dtype=torch.long)
        eval_mask = torch.tensor(all_attention_mask_np[:eval_size], dtype=torch.long)
        eval_topk_v = torch.tensor(topk_values_np[:eval_size].astype(np.float32))
        eval_topk_i = torch.tensor(topk_indices_np[:eval_size].astype(np.int64))

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

                    input_ids = torch.tensor(all_input_ids_np[mini_idx], dtype=torch.long)
                    attention_mask = torch.tensor(all_attention_mask_np[mini_idx], dtype=torch.long)
                    t_topk_v = torch.tensor(topk_values_np[mini_idx].astype(np.float32))
                    t_topk_i = torch.tensor(topk_indices_np[mini_idx].astype(np.int64))

                    s_out = model(input_ids=input_ids, attention_mask=attention_mask)
                    s_logits = s_out.logits

                    s_log_probs = F.log_softmax(s_logits / args.kd_temp, dim=-1)
                    s_log_probs_topk = s_log_probs.gather(-1, t_topk_i)
                    t_probs = F.softmax(t_topk_v / args.kd_temp, dim=-1)
                    pad_mask = attention_mask.unsqueeze(-1).float()
                    kl = (t_probs * (torch.log(t_probs + 1e-9) - s_log_probs_topk)) * pad_mask
                    kd = kl.sum(dim=-1).mean()

                    if ce_alpha > 0.0:
                        ce_mask = attention_mask[:, 1:].float()
                        ce_nll = F.cross_entropy(
                            s_logits[:, :-1].reshape(-1, s_logits.size(-1)),
                            input_ids[:, 1:].reshape(-1),
                            reduction="none",
                        ).reshape(input_ids[:, 1:].shape)
                        ce = (ce_nll * ce_mask).sum() / ce_mask.sum().clamp(min=1)
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
                        e_out = model(input_ids=eval_ids, attention_mask=eval_mask)
                        e_logits = e_out.logits
                        e_log_p = F.log_softmax(e_logits / args.kd_temp, dim=-1)
                        e_log_p_topk = e_log_p.gather(-1, eval_topk_i)
                        et_probs = F.softmax(eval_topk_v / args.kd_temp, dim=-1)
                        emask = eval_mask.unsqueeze(-1).float()
                        kl_eval = (et_probs * (torch.log(et_probs + 1e-9) - e_log_p_topk)) * emask
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
