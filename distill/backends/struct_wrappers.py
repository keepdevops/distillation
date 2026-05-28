"""Shared struct utilities — to_dict(), validation, dataclass conversion.

Used by all C++ bridge modules to convert between pybind11 structs,
plain dicts, and Python dataclasses without duplicating boilerplate.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def to_dict(obj: Any) -> dict[str, Any]:
    """Convert a pybind11 struct, dataclass, or plain dict to a dict.

    Tries .to_dict() first, then dataclass asdict(), then vars().
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    try:
        from dataclasses import asdict as _asdict
        return _asdict(obj)
    except TypeError:
        pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot convert {type(obj)} to dict")


def validate_thermal(d: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise a thermal reading dict."""
    float_keys = ["cpu_temp", "gpu_temp", "soc_temp", "cpu_power", "gpu_power", "total_power"]
    for k in float_keys:
        if k in d:
            try:
                d[k] = float(d[k])
            except (TypeError, ValueError) as exc:
                logger.warning("thermal field %s invalid: %s", k, exc)
                d[k] = 0.0
    d.setdefault("available", False)
    d.setdefault("error", "")
    return d


def validate_quant_config(d: dict[str, Any]) -> dict[str, Any]:
    """Validate a QuantConfig dict."""
    valid_methods = {"q4_k_m", "q5_k_m", "q8_0", "awq", "gptq", "exl2", "onnx", "mlx"}
    method = str(d.get("method", "q4_k_m")).lower()
    d["method"] = method   # persist normalised value
    if method not in valid_methods:
        logger.warning("Unknown quant method '%s', falling back to q4_k_m", method)
        d["method"] = "q4_k_m"
    bits = int(d.get("bits", 4))
    if bits not in (2, 3, 4, 5, 6, 8, 16):
        logger.warning("Unusual bits value %d", bits)
    d["bits"] = bits
    d.setdefault("group_size", 128)
    d.setdefault("output_format", "gguf")
    d.setdefault("output_path", "")
    return d


def validate_model_metrics(d: dict[str, Any]) -> dict[str, Any]:
    """Validate a ModelMetrics dict."""
    for k in ["tokens_per_sec", "ttft_ms", "peak_memory_gb", "perplexity", "quality_score"]:
        if k in d:
            try:
                d[k] = float(d[k])
            except (TypeError, ValueError):
                d[k] = 0.0
    if "param_count" in d:
        d["param_count"] = int(d["param_count"])
    d.setdefault("model_id", "")
    d.setdefault("backend", "unknown")
    return d


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple dicts left-to-right; later values win."""
    result: dict[str, Any] = {}
    for d in dicts:
        result.update(d)
    return result


def dataclass_from_dict(cls: type[T], d: dict[str, Any]) -> T:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in d.items() if k in known})
