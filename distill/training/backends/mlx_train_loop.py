"""Epoch/step training loop for MLX knowledge distillation.

Handles gradient accumulation, logging, eval, checkpointing, and
pause-flag monitoring. Called by mlx.main() after all setup is complete.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import numpy as np

from ...infra.train_utils import check_pause_flag, write_metric
from .mlx_grad_utils import add_grads as _add_grads, scale_grads as _scale_grads, grad_norm as _grad_norm
from .mlx_trainer_state import write_trainer_state

LOG = logging.getLogger(__name__)


def run_training_loop(
    args,
    student_model,
    optimizer,
    all_input_ids: np.ndarray,
    all_attention_mask: np.ndarray,
    all_teacher_topk_values: np.ndarray,
    all_teacher_topk_indices: np.ndarray,
    eval_tensors: tuple,
    loss_and_grad,
    kd_loss_fn,
    current_temp_fn,
    current_alpha_fn,
    metrics_path: Path,
    trainer_state_path: Path,
    output_dir: Path,
    total_steps: int,
    steps_per_epoch: int,
    start_epoch: int = 0,
    start_global_step: int = 0,
) -> None:
    """Run the full MLX KD training loop.

    Args:
        args: Parsed argparse namespace with training hyperparameters.
        student_model: MLX student model with LoRA applied, in train mode.
        optimizer: MLX AdamW optimizer with LR schedule already configured.
        all_input_ids: (N, T) int32 numpy array of pre-tokenized IDs.
        all_attention_mask: (N, T) int32 numpy array of attention masks.
        all_teacher_topk_values: (N, T, K) float16 cached teacher logits.
        all_teacher_topk_indices: (N, T, K) int32 cached teacher top-K indices.
        eval_tensors: Tuple of (eval_ids, eval_mask, eval_topk_values, eval_topk_indices)
            as MLX arrays for eval micro-batching.
        loss_and_grad: nn.value_and_grad-wrapped loss function.
        kd_loss_fn: Raw kd_loss callable (for eval without gradient).
        current_temp_fn: Callable(step) -> float KD temperature.
        current_alpha_fn: Callable(step) -> float CE alpha weight.
        metrics_path: Path to metrics.jsonl output file.
        trainer_state_path: Path to trainer_state.json for watchdog.
        output_dir: Root output directory (used for pause.flag and checkpoints).
        total_steps: Pre-computed total optimizer steps across all epochs.
        steps_per_epoch: Pre-computed optimizer steps per epoch.
        start_epoch: Epoch index to resume from (0 for fresh run).
        start_global_step: Global step counter to resume from (0 for fresh).
    """
    import mlx.core as mx

    eval_ids, eval_mask, eval_topk_values, eval_topk_indices = eval_tensors
    eval_size = eval_ids.shape[0]

    batch_size = args.batch_size
    grad_acc = args.grad_acc
    macro_batch = batch_size * grad_acc
    n_samples = all_input_ids.shape[0]
    checkpoint_path = output_dir / "checkpoint.json"

    t0 = time.time()
    log_history: list[dict] = []
    global_step = start_global_step

    for epoch in range(start_epoch, args.epochs):
        indices = list(range(n_samples))
        random.shuffle(indices)

        for step_start in range(0, n_samples - macro_batch + 1, macro_batch):
            if args.watchdog and check_pause_flag(output_dir):
                LOG.info("Saving student weights before exit.")
                try:
                    student_model.save_weights(str(output_dir / "mlx_student_weights.npz"))
                except Exception as e:
                    LOG.error("Failed to save weights on pause: %s", e)
                    raise
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
                    current_temp_fn(global_step), current_alpha_fn(global_step),
                )
                mx.eval(loss_val, grads)
                accum_loss += float(loss_val) / grad_acc
                accum_grads = grads if accum_grads is None else _add_grads(accum_grads, grads)
                mx.eval(accum_grads)
                # Free intermediate activations (s_logits, log_probs) after each micro-batch
                mx.clear_cache()

            accum_grads = _scale_grads(accum_grads, 1.0 / grad_acc)
            gn = _grad_norm(accum_grads)  # float() inside forces eval before grads are freed
            optimizer.update(student_model, accum_grads)
            mx.eval(student_model.parameters(), optimizer.state)
            mx.clear_cache()  # free optimizer intermediates after each optimizer step

            global_step += 1
            epoch_frac = epoch + step_start / n_samples

            if global_step % args.log_steps == 0:
                elapsed = time.time() - t0
                LOG.info(
                    "step=%d  epoch=%.2f  loss=%.4f  grad_norm=%.4f  %.2f steps/s",
                    global_step, epoch_frac, accum_loss, gn,
                    global_step / max(elapsed, 1e-6),
                )
                write_metric(metrics_path, global_step, epoch_frac, loss=accum_loss, grad_norm=gn)
                log_history.append({
                    "step": global_step, "epoch": epoch_frac,
                    "loss": accum_loss, "grad_norm": gn,
                })
                write_trainer_state(trainer_state_path, log_history)

            if global_step % args.eval_steps == 0:
                _run_eval(
                    student_model, kd_loss_fn,
                    eval_ids, eval_mask, eval_topk_values, eval_topk_indices,
                    eval_size, batch_size,
                    current_temp_fn(global_step), current_alpha_fn(global_step),
                    global_step, epoch_frac,
                    metrics_path, trainer_state_path, log_history,
                )

            if global_step >= total_steps:
                break

        # Save epoch checkpoint (enables --resume on crash/interrupt)
        try:
            student_model.save_weights(str(output_dir / "mlx_student_weights.npz"))
            import json
            with open(checkpoint_path, "w") as f:
                json.dump({"epoch": epoch, "global_step": global_step}, f)
            LOG.info("Checkpoint saved: epoch=%d  global_step=%d", epoch, global_step)
        except Exception as e:
            LOG.error("Failed to save epoch checkpoint at epoch=%d: %s", epoch, e)
            raise

        if global_step >= total_steps:
            break


def _run_eval(
    student_model,
    kd_loss_fn,
    eval_ids,
    eval_mask,
    eval_topk_values,
    eval_topk_indices,
    eval_size: int,
    batch_size: int,
    kd_temp: float,
    ce_alpha: float,
    global_step: int,
    epoch_frac: float,
    metrics_path: Path,
    trainer_state_path: Path,
    log_history: list,
) -> None:
    """Run eval micro-batches and log eval_loss.

    Runs without gradient computation to avoid ~20 GB activation buffers
    that loss_and_grad would allocate for eval_size=32 samples.
    """
    import mlx.core as mx

    eval_losses: list[float] = []
    try:
        for _eb in range(0, eval_size, batch_size):
            _eids = eval_ids[_eb: _eb + batch_size]
            _emsk = eval_mask[_eb: _eb + batch_size]
            _etv  = eval_topk_values[_eb: _eb + batch_size]
            _eti  = eval_topk_indices[_eb: _eb + batch_size]
            _el = kd_loss_fn(student_model, _eids, _emsk, _etv, _eti, kd_temp, ce_alpha)
            mx.eval(_el)
            eval_losses.append(float(_el))
            mx.clear_cache()
    except Exception as e:
        LOG.error("Eval micro-batch failed at step=%d: %s", global_step, e)
        raise

    eval_loss_val = sum(eval_losses) / len(eval_losses)
    LOG.info("  eval_loss=%.4f", eval_loss_val)
    write_metric(metrics_path, global_step, epoch_frac, eval_loss=eval_loss_val)
    log_history.append({"step": global_step, "epoch": epoch_frac, "eval_loss": eval_loss_val})
    write_trainer_state(trainer_state_path, log_history)
