"""Python adapter for QuantConfig and ModelMetrics C++ structs.

Bridges between UI config dicts and the distill_cpp pybind11 structs.
Falls back to plain dicts when the C++ module is not compiled.
"""
from __future__ import annotations

import logging
from typing import Any

from distill.backends.struct_wrappers import validate_quant_config, validate_model_metrics

logger = logging.getLogger(__name__)

_QUANT_PRESETS: dict[str, dict[str, Any]] = {
    "q4_k_m": {"method": "q4_k_m", "bits": 4, "group_size": 128,
               "output_format": "gguf", "use_k_quant": True},
    "q5_k_m": {"method": "q5_k_m", "bits": 5, "group_size": 128,
               "output_format": "gguf", "use_k_quant": True},
    "q8_0":   {"method": "q8_0",   "bits": 8, "group_size": 32,
               "output_format": "gguf", "use_k_quant": False},
    "awq":    {"method": "awq",    "bits": 4, "group_size": 128,
               "output_format": "safetensors", "use_k_quant": False},
    "gptq":   {"method": "gptq",   "bits": 4, "group_size": 128,
               "output_format": "safetensors", "use_k_quant": False},
    "exl2":   {"method": "exl2",   "bits": 4, "group_size": 64,
               "output_format": "exl2", "use_k_quant": False},
}


def _try_cpp():
    try:
        import distill_cpp  # type: ignore[import]
        return distill_cpp.QuantConfig, distill_cpp.ModelMetrics
    except ImportError:
        return None, None


def build_quant_config(
    method: str = "q4_k_m",
    output_path: str = "",
    perplexity_threshold: float = 0.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Return a validated QuantConfig dict, using C++ struct when available."""
    base = _QUANT_PRESETS.get(method.lower(), _QUANT_PRESETS["q4_k_m"]).copy()
    base.update({"output_path": output_path,
                 "perplexity_threshold": perplexity_threshold, **overrides})
    base = validate_quant_config(base)

    QuantConfig, _ = _try_cpp()
    if QuantConfig is not None:
        try:
            q = QuantConfig()
            q.method               = base["method"]
            q.bits                 = base["bits"]
            q.group_size           = base["group_size"]
            q.use_k_quant          = base.get("use_k_quant", False)
            q.perplexity_threshold = base["perplexity_threshold"]
            q.output_format        = base["output_format"]
            q.output_path          = base["output_path"]
            return {**q.to_dict(), "cpp_backed": True}
        except Exception as exc:
            logger.warning("C++ QuantConfig failed: %s", exc)

    return {**base, "cpp_backed": False}


def build_model_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Return a validated ModelMetrics dict, using C++ struct when available."""
    data = validate_model_metrics(dict(data))
    _, ModelMetrics = _try_cpp()

    if ModelMetrics is not None:
        try:
            m = ModelMetrics()
            m.model_id        = data.get("model_id", "")
            m.backend         = data.get("backend", "")
            m.tokens_per_sec  = float(data.get("tokens_per_sec", 0.0))
            m.ttft_ms         = float(data.get("ttft_ms", 0.0))
            m.peak_memory_gb  = float(data.get("peak_memory_gb", 0.0))
            m.param_count     = int(data.get("param_count", 0))
            m.perplexity      = float(data.get("perplexity", 0.0))
            m.quality_score   = float(data.get("quality_score", 0.0))
            return {**m.to_dict(), "cpp_backed": True}
        except Exception as exc:
            logger.warning("C++ ModelMetrics failed: %s", exc)

    return {**data, "cpp_backed": False}


def compare_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, str]:
    """Compare two ModelMetrics dicts. Returns per-field winner ('a', 'b', 'tie')."""
    _, ModelMetrics = _try_cpp()
    if ModelMetrics is not None:
        try:
            ma, mb = ModelMetrics(), ModelMetrics()
            for m, d in [(ma, a), (mb, b)]:
                m.perplexity   = float(d.get("perplexity", 0.0))
                m.quality_score = float(d.get("quality_score", 0.0))
            winner = "a" if ma.is_better_than(mb) else "b"
            return {"overall": winner}
        except Exception:
            pass

    result: dict[str, str] = {}
    for key, lower_better in [("perplexity", True), ("quality_score", False),
                               ("tokens_per_sec", False)]:
        av, bv = float(a.get(key, 0)), float(b.get(key, 0))
        if av == bv:
            result[key] = "tie"
        elif lower_better:
            result[key] = "a" if av < bv else "b"
        else:
            result[key] = "a" if av > bv else "b"
    return result


def quant_preset_names() -> list[str]:
    return list(_QUANT_PRESETS.keys())
