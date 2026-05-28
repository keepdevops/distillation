"""Graceful OOM recovery — auto-halve batch size and retry on memory errors.

Wraps a training function with exponential back-off: on CUDA/MPS OOM,
batch_size is halved and grad_accum doubled (preserving effective batch size),
then the run is retried up to max_retries times.
"""
from __future__ import annotations

import gc
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

_OOM_SIGNALS = (
    "CUDA out of memory",
    "MPS backend ran out of memory",
    "out of memory",
    "Cannot allocate memory",
    "metal: error",
)

_MIN_BATCH_SIZE = 1
_MAX_RETRIES    = 4


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s.lower() in msg for s in _OOM_SIGNALS)


def _free_memory() -> None:
    """Aggressively free GPU/MPS memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def with_oom_recovery(
    train_fn: Callable[..., Any],
    train_kwargs: dict[str, Any],
    batch_size_key: str = "batch_size",
    grad_accum_key: str = "grad_accum",
    max_retries: int = _MAX_RETRIES,
) -> Any:
    """Call train_fn(**train_kwargs), retrying with smaller batch on OOM.

    On each OOM:
      - batch_size is halved (floor, minimum 1)
      - grad_accum is doubled (preserving effective batch size)
      - memory is freed before retry

    Args:
        train_fn:        The training function to call.
        train_kwargs:    Keyword arguments passed to train_fn (mutated on retry).
        batch_size_key:  Key in train_kwargs for per-device batch size.
        grad_accum_key:  Key in train_kwargs for gradient accumulation steps.
        max_retries:     Maximum number of OOM retries.

    Returns:
        Whatever train_fn returns on success.

    Raises:
        RuntimeError: If max_retries is exhausted or a non-OOM exception occurs.
    """
    kwargs = dict(train_kwargs)
    attempt = 0

    while attempt <= max_retries:
        bs = kwargs.get(batch_size_key, 1)
        ga = kwargs.get(grad_accum_key, 1)

        try:
            logger.info(
                "Training attempt %d/%d (batch_size=%d, grad_accum=%d)",
                attempt + 1, max_retries + 1, bs, ga,
            )
            return train_fn(**kwargs)

        except Exception as exc:
            if not _is_oom(exc):
                raise  # re-raise non-OOM exceptions immediately

            attempt += 1
            new_bs = max(_MIN_BATCH_SIZE, bs // 2)
            new_ga = ga * (bs // new_bs) if new_bs < bs else ga * 2

            if attempt > max_retries:
                raise RuntimeError(
                    f"OOM persists after {max_retries} retries "
                    f"(final batch_size={bs}). "
                    "Try reducing lora_rank, max_length, or model size."
                ) from exc

            logger.warning(
                "OOM on attempt %d: %s\n"
                "  Reducing %s: %d → %d, %s: %d → %d and retrying...",
                attempt, exc,
                batch_size_key, bs, new_bs,
                grad_accum_key, ga, new_ga,
            )
            _free_memory()
            kwargs[batch_size_key] = new_bs
            kwargs[grad_accum_key] = new_ga

    raise RuntimeError("OOM recovery exhausted all retries")


def safe_train(
    train_fn: Callable[..., Any],
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Convenience wrapper: returns (result, final_kwargs_used)."""
    final_kwargs = dict(kwargs)
    result = with_oom_recovery(train_fn, final_kwargs)
    return result, final_kwargs
