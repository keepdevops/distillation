"""Metal memory utilities for MLX teacher logit precompute.

Provides a safe batch-size calculation that accounts for current model
occupancy so precompute doesn't OOM on the first batch.
"""
from __future__ import annotations

import logging

LOG = logging.getLogger(__name__)


def memory_safe_precomp_bs(requested_bs: int, seq_len: int, vocab_size: int) -> int:
    """Return a precomp_bs that fits in Metal memory given current model occupancy.

    After the teacher is loaded, measure how much Metal memory it occupies, then
    budget the remainder for the logits + argsort temporaries (peak ~3x one logits
    tensor: input logits + sort indices + sorted values).

    Args:
        requested_bs: The desired batch size before memory capping.
        seq_len: Token sequence length (from pre-tokenized dataset).
        vocab_size: Teacher model vocabulary size.

    Returns:
        An integer batch size <= requested_bs that should fit in Metal memory.
    """
    try:
        import mlx.core as mx
        info = mx.metal.device_info()
        # recommendedMaxWorkingSetSize is the OS-blessed working-set ceiling.
        # Fall back to 75% of total RAM if not reported.
        recommended = info.get(
            "recommendedMaxWorkingSetSize",
            int(info.get("memorySize", 0) * 0.75),
        )
        if recommended <= 0:
            return requested_bs

        # How much is already occupied by the loaded teacher model?
        active_bytes = mx.metal.get_active_memory()   # returns int bytes
        headroom = recommended - active_bytes
        if headroom <= 0:
            LOG.warning(
                "Metal memory already at/above recommended limit before precompute "
                "(active=%.1f GB, limit=%.1f GB). Using precomp_bs=1.",
                active_bytes / 1e9, recommended / 1e9,
            )
            return 1

        # Each logit batch: B x T x V x 4 bytes (float32).
        # argsort creates a same-shape index tensor + a same-shape sorted-values tensor,
        # so peak Metal allocation is ~3x one logit tensor.
        bytes_per_sample = seq_len * vocab_size * 4
        # Use 50% of headroom to leave room for stack/scratch; argsort peaks at 3x.
        max_bs = max(1, int(headroom * 0.50) // (bytes_per_sample * 3))
        if max_bs < requested_bs:
            LOG.warning(
                "Reducing precomp_bs %d → %d to fit in Metal memory "
                "(model=%.1f GB, headroom=%.1f GB, peak/batch=%.2f GB, vocab=%d, seq=%d).",
                requested_bs, max_bs,
                active_bytes / 1e9, headroom / 1e9,
                bytes_per_sample * 3 * requested_bs / 1e9,
                vocab_size, seq_len,
            )
        return min(requested_bs, max_bs)
    except Exception as e:
        LOG.debug("Memory probe failed (%s); using requested precomp_bs=%d.", e, requested_bs)
        return requested_bs
