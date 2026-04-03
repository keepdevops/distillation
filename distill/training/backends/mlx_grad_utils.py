"""Gradient tree utilities for MLX training (add, scale, norm)."""
from __future__ import annotations


def add_grads(a, b):
    """Recursively add two gradient trees (nested dicts/lists of mx.arrays)."""
    import mlx.core as mx
    if isinstance(a, mx.array):
        return a + b
    if isinstance(a, dict):
        return {k: add_grads(a[k], b[k]) for k in a}
    if isinstance(a, list):
        return [add_grads(x, y) for x, y in zip(a, b)]
    return a


def scale_grads(g, scale):
    """Recursively scale a gradient tree by a scalar."""
    import mlx.core as mx
    if isinstance(g, mx.array):
        return g * scale
    if isinstance(g, dict):
        return {k: scale_grads(v, scale) for k, v in g.items()}
    if isinstance(g, list):
        return [scale_grads(x, scale) for x in g]
    return g


def grad_norm(g) -> float:
    """Compute global L2 gradient norm across all tensors in the gradient tree."""
    import mlx.core as mx
    sq_sums: list = []

    def _collect(x):
        if isinstance(x, mx.array):
            sq_sums.append(mx.sum(x * x))
        elif isinstance(x, dict):
            for v in x.values():
                _collect(v)
        elif isinstance(x, list):
            for v in x:
                _collect(v)

    _collect(g)
    if not sq_sums:
        return 0.0
    return float(mx.sqrt(mx.sum(mx.stack(sq_sums))))
