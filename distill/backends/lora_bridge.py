"""Python adapter for LoRAConfig and LoRATrainingMetrics C++ structs.

Falls back to distill.config.schemas.LoRAConfig (Pydantic) when distill_cpp
is not compiled. Both paths produce the same dict shape.
"""
from __future__ import annotations

import logging
from typing import Any

from distill.config.schemas import LoRAConfig as PyLoRAConfig
from distill.backends.struct_wrappers import validate_thermal

logger = logging.getLogger(__name__)


def _try_cpp_lora():
    try:
        import distill_cpp  # type: ignore[import]
        return distill_cpp.LoRAConfig, distill_cpp.LoRATrainingMetrics
    except (ImportError, AttributeError):
        return None, None


def build_lora_config(
    rank: int = 16,
    alpha: int | None = None,
    dropout: float = 0.05,
    use_qlora: bool = False,
    qlora_bits: int = 4,
    target_modules: list[str] | None = None,
    bias: str = "none",
) -> dict[str, Any]:
    """Return a validated LoRAConfig dict, using C++ struct when available.

    Alpha defaults to 2× rank if not specified (standard practice).
    """
    if alpha is None:
        alpha = rank * 2
    targets = target_modules or ["q_proj", "v_proj"]

    CppLoRA, _ = _try_cpp_lora()
    if CppLoRA is not None:
        try:
            c = CppLoRA()
            c.rank           = rank
            c.alpha          = alpha
            c.dropout        = dropout
            c.use_qlora      = use_qlora
            c.qlora_bits     = qlora_bits
            c.target_modules = targets
            c.bias           = bias
            return {**c.to_dict(), "cpp_backed": True}
        except Exception as exc:
            logger.warning("C++ LoRAConfig failed: %s — falling back", exc)

    # Pure-Python Pydantic path
    py = PyLoRAConfig(
        rank=rank, alpha=alpha, dropout=dropout,
        use_qlora=use_qlora, qlora_bits=qlora_bits,
        target_modules=targets, bias=bias,
    )
    return {
        "rank":           py.rank,
        "alpha":          py.alpha,
        "dropout":        py.dropout,
        "use_qlora":      py.use_qlora,
        "qlora_bits":     py.qlora_bits,
        "target_modules": py.target_modules,
        "bias":           py.bias,
        "scaling":        py.scaling,
        "cpp_backed":     False,
    }


def estimate_vram(
    rank: int,
    hidden_size: int = 2048,
    num_layers: int = 24,
    num_targets: int = 2,
    base_model_gb: float = 3.0,
    batch_size: int = 4,
    seq_len: int = 512,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Estimate total VRAM requirement for a LoRA training run (GB).

    Returns dict with adapter_mb, activations_mb, base_model_gb, total_gb.
    """
    # LoRA adapter params: 2 × hidden × rank × num_targets × num_layers
    adapter_params = 2 * hidden_size * rank * num_targets * num_layers
    adapter_mb = (adapter_params * dtype_bytes) / (1024 ** 2)

    # Activations: batch × seq × hidden × num_layers × dtype
    act_bytes = batch_size * seq_len * hidden_size * num_layers * dtype_bytes
    act_mb = act_bytes / (1024 ** 2)

    # Gradient + optimizer states ≈ 2× adapter params
    opt_mb = adapter_mb * 2.0

    total_gb = base_model_gb + (adapter_mb + act_mb + opt_mb) / 1024

    return {
        "adapter_mb":    round(adapter_mb, 1),
        "activations_mb": round(act_mb, 1),
        "optimizer_mb":  round(opt_mb, 1),
        "base_model_gb": round(base_model_gb, 1),
        "total_gb":      round(total_gb, 2),
    }


def push_training_metrics(
    step: int,
    adapter_norms: list[float] | None = None,
    update_ratios: list[float] | None = None,
    grad_norms: list[float] | None = None,
) -> dict[str, Any]:
    """Build a LoRATrainingMetrics dict from raw per-layer arrays.

    Uses C++ struct when available to call compute_aggregates().
    """
    _, CppMetrics = _try_cpp_lora()
    norms   = adapter_norms  or []
    ratios  = update_ratios  or []
    grads   = grad_norms     or []

    if CppMetrics is not None:
        try:
            import distill_cpp
            m = CppMetrics()
            m.step = step
            for i, (norm, ratio, grad) in enumerate(zip(norms, ratios, grads)):
                layer = distill_cpp.LoRALayerMetrics()
                layer.layer_name   = f"layer_{i}"
                layer.adapter_norm = norm
                layer.update_ratio = ratio
                layer.grad_norm    = grad
                layer.is_active    = norm > 1e-8
                m.push_layer(layer)
            m.compute_aggregates()
            return {**m.to_dict(), "cpp_backed": True}
        except Exception as exc:
            logger.warning("C++ LoRATrainingMetrics failed: %s", exc)

    # Python fallback
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "step":              step,
        "mean_adapter_norm": mean(norms),
        "mean_update_ratio": mean(ratios),
        "mean_grad_norm":    mean(grads),
        "rank_utilization":  1.0,
        "dead_layers":       sum(1 for n in norms if n < 1e-8),
        "health_ok":         all(n > 1e-8 for n in norms) if norms else True,
        "cpp_backed":        False,
    }
