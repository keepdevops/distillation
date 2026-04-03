"""OOM-safe teacher logit precomputation for MLX distillation.

Runs the frozen teacher over the entire dataset once and caches the top-K
logit values and indices. Storing top-K=50 indices+values uses ~300 MB vs
311 GB for a full-vocab float32 cache.
"""
from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger(__name__)


def precompute_teacher_logits(
    teacher_model,
    all_input_ids: np.ndarray,
    n_samples: int,
    seq_len: int,
    K: int,
    precomp_bs: int,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute teacher top-K logits for all samples.

    Teacher is frozen and the dataset is fixed, so we compute once and reuse
    every epoch. An OOM-safe retry loop halves precomp_bs on Metal OOM errors
    and restarts from scratch.

    Args:
        teacher_model: Frozen MLX teacher model (already loaded).
        all_input_ids: (N, T) int32 numpy array of pre-tokenized input IDs.
        n_samples: Number of training samples (== all_input_ids.shape[0]).
        seq_len: Sequence length T (== all_input_ids.shape[1]).
        K: Number of top-K logits to keep per token position.
        precomp_bs: Initial batch size for teacher forward passes.
        logger: Optional logger; falls back to module-level LOG if None.

    Returns:
        Tuple of:
          - all_teacher_topk_values:   (N, T, K) float16 numpy array
          - all_teacher_topk_indices:  (N, T, K) int32 numpy array
    """
    import mlx.core as mx

    _log = logger or LOG

    all_teacher_topk_values = np.zeros((n_samples, seq_len, K), dtype=np.float16)
    all_teacher_topk_indices = np.zeros((n_samples, seq_len, K), dtype=np.int32)

    # OOM-safe loop: halve precomp_bs and restart on Metal out-of-memory errors.
    while precomp_bs >= 1:
        try:
            for precomp_start in range(0, n_samples, precomp_bs):
                precomp_end = min(precomp_start + precomp_bs, n_samples)
                batch_ids = mx.array(all_input_ids[precomp_start:precomp_end])
                t_out = teacher_model(batch_ids)
                t_logits = t_out if isinstance(t_out, mx.array) else t_out.logits  # (B, T, V)
                mx.eval(t_logits)
                topk_idx = mx.argsort(-t_logits, axis=-1)[..., :K]            # (B, T, K)
                topk_val = mx.take_along_axis(t_logits, topk_idx, axis=-1)    # (B, T, K)
                mx.eval(topk_idx, topk_val)
                all_teacher_topk_values[precomp_start:precomp_end] = (
                    np.array(topk_val.astype(mx.float32)).astype(np.float16)
                )
                all_teacher_topk_indices[precomp_start:precomp_end] = np.array(
                    topk_idx.astype(mx.int32)
                )
                mx.clear_cache()  # free MLX intermediate buffers after each batch
                if precomp_end % max(precomp_bs * 10, 1) == 0 or precomp_end == n_samples:
                    _log.info("  Teacher logits: %d/%d samples", precomp_end, n_samples)
            break  # completed successfully
        except RuntimeError as e:
            if (
                "OutOfMemory" in str(e)
                or "Insufficient Memory" in str(e)
                or "kIOGPUCommandBuffer" in str(e)
            ):
                mx.clear_cache()
                new_bs = max(1, precomp_bs // 2)
                _log.warning(
                    "Metal OOM at precomp_bs=%d — retrying with precomp_bs=%d.",
                    precomp_bs, new_bs,
                )
                precomp_bs = new_bs
                # Reset partially-filled output arrays before retry
                all_teacher_topk_values[:] = 0
                all_teacher_topk_indices[:] = 0
            else:
                _log.error("Non-OOM RuntimeError during teacher precompute: %s", e)
                raise

    cache_mb = (all_teacher_topk_values.nbytes + all_teacher_topk_indices.nbytes) / 1e6
    _log.info("Teacher top-%d logits cached (%.0f MB).", K, cache_mb)

    return all_teacher_topk_values, all_teacher_topk_indices
