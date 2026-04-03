"""KD loss function for MLX distillation.

Mixed loss: ce_alpha * cross-entropy + (1 - ce_alpha) * forward-KL divergence
over the top-K teacher logit sparse distribution.

kd_temp and ce_alpha are scalar arguments (not closed-over) so that
nn.value_and_grad treats them as constants during the backward pass while
still allowing per-step annealing at the call site.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def kd_loss(
    model,
    input_ids,
    attention_mask,
    t_topk_values,
    t_topk_indices,
    kd_temp: float,
    ce_alpha: float,
):
    """Mixed knowledge-distillation loss (forward KL + optional CE).

    Args:
        model: Student MLX model in training mode.
        input_ids: (B, T) int32 MLX array of token IDs.
        attention_mask: (B, T) int32 MLX array (1 = real token, 0 = pad).
        t_topk_values: (B, T, K) float32 MLX array — raw teacher logits at
            top-K vocab positions.
        t_topk_indices: (B, T, K) int32 MLX array — vocab indices of the K
            teacher top-logit positions.
        kd_temp: KD temperature scalar (supports per-step annealing).
        ce_alpha: CE weight scalar in [0, 1] (supports per-step annealing).
            0 = pure KD, 1 = pure CE.

    Returns:
        Scalar loss value (MLX array).
    """
    s_out = model(input_ids)
    s_logits = s_out if isinstance(s_out, mx.array) else s_out.logits        # (B, T, V)

    # ── KD term: forward KL over top-K sparse teacher distribution ───────────
    s_log_probs = nn.log_softmax(s_logits / kd_temp, axis=-1)                # (B, T, V)
    s_log_probs_topk = mx.take_along_axis(s_log_probs, t_topk_indices, axis=-1)  # (B, T, K)
    t_probs = nn.softmax(t_topk_values / kd_temp, axis=-1)                   # (B, T, K)
    mask = attention_mask[..., None].astype(mx.float32)                       # (B, T, 1)
    kl = (t_probs * (mx.log(t_probs + 1e-9) - s_log_probs_topk)) * mask
    kd = kl.sum(axis=-1).mean()

    if ce_alpha == 0.0:
        return kd

    # ── CE term: next-token prediction ───────────────────────────────────────
    ce_logits  = s_logits[:, :-1]                                             # (B, T-1, V)
    ce_targets = input_ids[:, 1:][..., None]                                  # (B, T-1, 1)
    ce_mask    = attention_mask[:, 1:].astype(mx.float32)                     # (B, T-1)
    ce_log_p   = nn.log_softmax(ce_logits, axis=-1)                           # (B, T-1, V)
    ce_nll     = -mx.take_along_axis(ce_log_p, ce_targets, axis=-1).squeeze(-1)  # (B, T-1)
    ce = (ce_nll * ce_mask).sum() / mx.maximum(ce_mask.sum(), 1.0)

    return ce_alpha * ce + (1.0 - ce_alpha) * kd
