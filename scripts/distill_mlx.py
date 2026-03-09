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

from data_pipeline import load_dataset_split, format_prompt_full, format_multiturn_full, pretokenize, validate_dataset_schema, DATASET_HELP

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

OPEN_TEACHER = "Qwen/Qwen2-1.5B-Instruct"
OPEN_STUDENT = "Qwen/Qwen2-0.5B-Instruct"

# Local MLX-converted model paths (pre-converted once via mlx_lm.convert, loads instantly)
_MLX_LOCAL_PATHS = {
    "Qwen/Qwen2-1.5B-Instruct": "airgap_bundle/mlx_models/qwen2-1.5b-instruct",
    "Qwen/Qwen2-0.5B-Instruct": "airgap_bundle/mlx_models/qwen2-0.5b-instruct",
    "meta-llama/Llama-3.2-8B-Instruct": "airgap_bundle/mlx_models/llama-3.2-8b-instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "airgap_bundle/mlx_models/llama-3.2-1b-instruct",
}


def _resolve_mlx_path(model_id: str) -> str:
    """Return local MLX path if pre-converted, otherwise the original HF model ID."""
    rel = _MLX_LOCAL_PATHS.get(model_id)
    if rel:
        # Check relative to cwd and relative to this script's repo root
        for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
            candidate = base / rel
            if candidate.exists() and any(candidate.iterdir()):
                LOG.info("Using local MLX model: %s", candidate)
                return str(candidate)
    return model_id


def parse_args():
    p = argparse.ArgumentParser(description="MLX knowledge distillation")
    p.add_argument("--teacher", type=str, default="meta-llama/Llama-3.2-8B-Instruct")
    p.add_argument("--student", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--open", action="store_true", help="Use open Qwen2 models (no HF login)")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help=DATASET_HELP)
    p.add_argument("--output_dir", type=str, default="./distilled-mlx")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2; student full-vocab logits are ~0.6 GB/sample at seq_len=512)")
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
    p.add_argument("--multi_turn_ratio", type=float, default=0.0,
                   help="Fraction of samples formatted as full multi-turn ChatML conversations "
                        "(0.0 = all single-turn, 1.0 = all multi-turn). "
                        "Only effective for ShareGPT/messages datasets. (default: 0.0)")
    p.add_argument("--precomp_bs", type=int, default=0,
                   help="Batch size for teacher logit pre-computation. No gradients are needed "
                        "so this can be much larger than --batch_size. 0 = auto (4× batch_size, "
                        "minimum 16). Larger values reduce Metal dispatch overhead. (default: 0)")
    # ── Phase 3: temperature annealing ───────────────────────────────────────────
    p.add_argument("--temp_start", type=float, default=0.0,
                   help="KD temperature at step 0. If >0 and != temp_end, linearly anneal from "
                        "temp_start → temp_end over all training steps. 0 = use --kd_temp fixed. "
                        "(default: 0, i.e. no annealing)")
    p.add_argument("--temp_end", type=float, default=0.0,
                   help="KD temperature at final step. Only used when temp_start > 0. (default: 0)")
    p.add_argument("--hard_weight_start", type=float, default=-1.0,
                   help="CE alpha (hard-label weight) at step 0. If >=0, linearly anneal from "
                        "hard_weight_start → hard_weight_end. -1 = use --ce_alpha fixed. "
                        "(default: -1, i.e. no annealing)")
    p.add_argument("--hard_weight_end", type=float, default=-1.0,
                   help="CE alpha at final step. Only used when hard_weight_start >=0. (default: -1)")
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

    # Ensure HuggingFace fast tokenizers (Rust/Rayon) use all available CPU threads.
    # Without this, HF may suppress parallelism with a warning when running from a subprocess.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

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

    # ── Load teacher only first (student loaded after teacher is freed) ──────────
    teacher_path = _resolve_mlx_path(args.teacher)
    LOG.info("Loading teacher: %s", teacher_path)
    teacher_model, teacher_tokenizer = mlx_load(teacher_path)
    teacher_model.freeze()
    LOG.info("Teacher loaded and frozen.")

    # ── Dataset ─────────────────────────────────────────────────────────────────
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

    # ── Pre-tokenize entire dataset once ────────────────────────────────────────
    # Uses teacher tokenizer (same vocab as student for same model family).
    LOG.info("Pre-tokenizing %d samples...", len(texts))
    all_input_ids, all_attention_mask = pretokenize(teacher_tokenizer, texts)
    n_samples = len(texts)
    seq_len = all_input_ids.shape[1]
    LOG.info("Pre-tokenization complete.")

    # ── Pre-compute teacher top-K logits for ALL samples ─────────────────────────
    # Teacher is frozen + dataset is fixed → compute once, reuse every epoch.
    # Storing top-K=50 indices+values uses ~300 MB vs 311 GB for the full vocab.
    K = args.topk_logits
    all_teacher_topk_values = np.zeros((n_samples, seq_len, K), dtype=np.float16)
    all_teacher_topk_indices = np.zeros((n_samples, seq_len, K), dtype=np.int32)

    # Use a larger batch for teacher precompute — no gradients means much lower memory pressure.
    # More samples per Metal dispatch → fewer CPU-GPU sync points → faster precompute.
    # Auto: 4× training batch size, min 16, max 32 (caps logits tensor at ~4 GB for 128k-vocab models)
    precomp_bs = args.precomp_bs if args.precomp_bs > 0 else min(max(args.batch_size * 4, 16), 32)
    LOG.info("Pre-computing teacher top-%d logits for %d samples (precomp_bs=%d)...",
             K, n_samples, precomp_bs)
    for precomp_start in range(0, n_samples, precomp_bs):
        precomp_end = min(precomp_start + precomp_bs, n_samples)
        batch_ids = mx.array(all_input_ids[precomp_start:precomp_end])
        t_out = teacher_model(batch_ids)
        t_logits = t_out if isinstance(t_out, mx.array) else t_out.logits  # (B, T, V)
        mx.eval(t_logits)
        topk_idx = mx.argsort(-t_logits, axis=-1)[..., :K]           # (B, T, K)
        topk_val = mx.take_along_axis(t_logits, topk_idx, axis=-1)   # (B, T, K)
        mx.eval(topk_idx, topk_val)
        all_teacher_topk_values[precomp_start:precomp_end] = np.array(topk_val.astype(mx.float32)).astype(np.float16)
        all_teacher_topk_indices[precomp_start:precomp_end] = np.array(topk_idx.astype(mx.int32))
        mx.clear_cache()  # free MLX intermediate buffers (logits, sort indices) after each batch
        if precomp_end % max(precomp_bs * 10, 1) == 0 or precomp_end == n_samples:
            LOG.info("  Teacher logits: %d/%d samples", precomp_end, n_samples)

    cache_mb = (all_teacher_topk_values.nbytes + all_teacher_topk_indices.nbytes) / 1e6
    LOG.info("Teacher top-%d logits cached (%.0f MB).", K, cache_mb)

    # Free teacher from memory — logits are fully cached, teacher not needed during training
    del teacher_model
    mx.clear_cache()

    # ── Load student AFTER teacher is freed (avoids both models in memory at once) ──
    student_path = _resolve_mlx_path(args.student)
    LOG.info("Loading student: %s", student_path)
    student_model, student_tokenizer = mlx_load(student_path)

    # Apply LoRA to student via mlx_lm helper (operates on last N transformer blocks)
    LOG.info("Applying LoRA (r=%d) to student...", args.lora_r)
    lora_config = {"rank": args.lora_r, "scale": 20.0, "dropout": 0.0}
    # Apply to all transformer blocks (num_layers=-1 means all)
    num_blocks = len(list(student_model.model.layers)) if hasattr(student_model, "model") else 24
    linear_to_lora_layers(student_model, num_blocks, lora_config)
    student_model.train()
    trainable = [(k, v) for k, v in student_model.trainable_parameters().items()]
    LOG.info("LoRA applied: %d trainable parameter tensors.", len(trainable))

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

    # ── Phase 3 annealing helpers ────────────────────────────────────────────────
    _do_temp_anneal = args.temp_start > 0.0 and args.temp_end > 0.0
    _do_alpha_anneal = args.hard_weight_start >= 0.0 and args.hard_weight_end >= 0.0

    def _anneal(start, end, step, total):
        return start + (end - start) * min(step / max(total, 1), 1.0)

    def _current_temp(step):
        if not _do_temp_anneal:
            return args.kd_temp
        return _anneal(args.temp_start, args.temp_end, step, _total_steps)

    def _current_alpha(step):
        if not _do_alpha_anneal:
            return args.ce_alpha
        return _anneal(args.hard_weight_start, args.hard_weight_end, step, _total_steps)

    if _do_temp_anneal:
        LOG.info("Temperature annealing: %.2f → %.2f over %d steps",
                 args.temp_start, args.temp_end, _total_steps)
    if _do_alpha_anneal:
        LOG.info("CE-alpha annealing: %.2f → %.2f over %d steps",
                 args.hard_weight_start, args.hard_weight_end, _total_steps)

    # ── KD loss: forward KL (top-K sparse) + optional CE ────────────────────────
    # kd_temp and ce_alpha are passed per-step to support Phase 3 annealing.
    # nn.value_and_grad differentiates w.r.t. model parameters; the extra args
    # (kd_temp, ce_alpha) are treated as constants during the backward pass.
    def kd_loss(model, input_ids, attention_mask, t_topk_values, t_topk_indices,
                kd_temp, ce_alpha):
        """Mixed loss: ce_alpha * CE + (1 - ce_alpha) * forward-KL.

        t_topk_values:  (B, T, K) float32 — raw teacher logits at top-K vocab positions
        t_topk_indices: (B, T, K) int32   — vocab indices of those K positions
        kd_temp:        scalar float — KD temperature (supports per-step annealing)
        ce_alpha:       scalar float — CE weight (supports per-step annealing)
        """
        s_out = model(input_ids)
        s_logits = s_out if isinstance(s_out, mx.array) else s_out.logits       # (B, T, V)

        # ── KD term ──────────────────────────────────────────────────────────────
        s_log_probs = nn.log_softmax(s_logits / kd_temp, axis=-1)               # (B, T, V)
        s_log_probs_topk = mx.take_along_axis(s_log_probs, t_topk_indices, axis=-1)  # (B, T, K)
        t_probs = nn.softmax(t_topk_values / kd_temp, axis=-1)                  # (B, T, K)
        mask = attention_mask[..., None].astype(mx.float32)                      # (B, T, 1)
        kl = (t_probs * (mx.log(t_probs + 1e-9) - s_log_probs_topk)) * mask
        kd = kl.sum(axis=-1).mean()

        if ce_alpha == 0.0:
            return kd

        # ── CE term: next-token prediction ───────────────────────────────────────
        ce_logits  = s_logits[:, :-1]                                            # (B, T-1, V)
        ce_targets = input_ids[:, 1:][..., None]                                 # (B, T-1, 1)
        ce_mask    = attention_mask[:, 1:].astype(mx.float32)                    # (B, T-1)
        ce_log_p   = nn.log_softmax(ce_logits, axis=-1)                          # (B, T-1, V)
        ce_nll     = -mx.take_along_axis(ce_log_p, ce_targets, axis=-1).squeeze(-1)  # (B, T-1)
        ce = (ce_nll * ce_mask).sum() / mx.maximum(ce_mask.sum(), 1.0)

        return ce_alpha * ce + (1.0 - ce_alpha) * kd

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
                    student_model, input_ids, attention_mask, t_topk_v, t_topk_i,
                    _current_temp(global_step), _current_alpha(global_step),
                )
                mx.eval(loss_val, grads)
                accum_loss += float(loss_val) / grad_acc
                accum_grads = grads if accum_grads is None else _add_grads(accum_grads, grads)
                mx.eval(accum_grads)
                mx.clear_cache()  # free intermediate activations (s_logits, log_probs) after each micro-batch

            accum_grads = _scale_grads(accum_grads, 1.0 / grad_acc)
            optimizer.update(student_model, accum_grads)
            mx.eval(student_model.parameters(), optimizer.state)
            mx.clear_cache()  # free optimizer intermediates after each optimizer step

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
                # Eval without gradient computation — avoids ~20 GB activation buffers
                # that loss_and_grad would allocate for eval_size=32 samples.
                # Run in micro-batches of batch_size to keep per-step peak memory small.
                eval_losses = []
                for _eb in range(0, eval_size, batch_size):
                    _eids = eval_ids[_eb: _eb + batch_size]
                    _emsk = eval_mask[_eb: _eb + batch_size]
                    _etv  = eval_topk_values[_eb: _eb + batch_size]
                    _eti  = eval_topk_indices[_eb: _eb + batch_size]
                    _el = kd_loss(student_model, _eids, _emsk, _etv, _eti,
                                  _current_temp(global_step), _current_alpha(global_step))
                    mx.eval(_el)
                    eval_losses.append(float(_el))
                    mx.clear_cache()
                eval_loss_val = sum(eval_losses) / len(eval_losses)
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
