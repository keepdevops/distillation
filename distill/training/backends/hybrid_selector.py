"""Hybrid backend auto-selection — choose the best training backend for the hardware.

Priority order:
  Apple Silicon (arm64): mlx > unsloth > sft > cpu
  NVIDIA GPU:            unsloth (if available) > sft > cpu
  CPU only:              sft (small models) > cpu

Returns a backend string compatible with distill.orchestration.agent --backend.
"""
from __future__ import annotations

import logging
import platform
from typing import Any

logger = logging.getLogger(__name__)


def _has_mlx() -> bool:
    try:
        import mlx.core  # type: ignore[import]
        return True
    except ImportError:
        return False


def _has_unsloth() -> bool:
    try:
        import unsloth  # type: ignore[import]
        return True
    except ImportError:
        return False


def _cuda_info() -> dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "available": True,
                "name": props.name,
                "vram_gb": props.total_memory / (1024 ** 3),
                "count": torch.cuda.device_count(),
            }
    except Exception:
        pass
    return {"available": False, "name": "", "vram_gb": 0.0, "count": 0}


def _mps_info() -> dict[str, Any]:
    try:
        import torch
        if torch.backends.mps.is_available():
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            return {"available": True, "ram_gb": ram_gb}
    except Exception:
        pass
    return {"available": False, "ram_gb": 0.0}


def select_backend(
    model_size_b: float = 1.0,
    prefer: str | None = None,
) -> dict[str, Any]:
    """Return the recommended backend and rationale.

    Args:
        model_size_b: Student model parameter count in billions.
        prefer:       User override (if set, validated and returned if feasible).

    Returns:
        dict with keys: backend, rationale, warnings, alternatives.
    """
    machine = platform.machine()
    warnings: list[str] = []
    alternatives: list[str] = []

    # ── Apple Silicon path ────────────────────────────────────────────────
    if machine == "arm64":
        mps = _mps_info()
        ram = mps["ram_gb"]

        if _has_mlx() and ram >= 8:
            backend = "mlx"
            rationale = f"Apple Silicon detected ({ram:.0f}GB unified RAM) — MLX is fastest."
            if model_size_b > ram * 0.6:
                warnings.append(
                    f"Model ({model_size_b:.1f}B params) may approach RAM limit ({ram:.0f}GB). "
                    "Consider reducing lora_rank or batch_size."
                )
            alternatives = ["sft", "unsloth"] if _has_unsloth() else ["sft"]
        elif _has_unsloth():
            backend = "unsloth"
            rationale = "MLX not available — using Unsloth on MPS."
            alternatives = ["sft"]
        else:
            backend = "sft"
            rationale = "Using standard SFT (HF Trainer) on MPS."
            alternatives = []

    # ── NVIDIA GPU path ───────────────────────────────────────────────────
    else:
        cuda = _cuda_info()
        if cuda["available"]:
            vram = cuda["vram_gb"]
            if _has_unsloth() and vram >= 8:
                backend = "unsloth"
                rationale = (
                    f"NVIDIA {cuda['name']} ({vram:.0f}GB VRAM) — "
                    "Unsloth provides 2-5× speedup with quantized training."
                )
                if model_size_b * 2 > vram:
                    warnings.append(
                        f"Model may exceed VRAM ({vram:.0f}GB). "
                        "Unsloth will use 4-bit loading automatically."
                    )
                alternatives = ["sft", "minillm"]
            elif vram >= 4:
                backend = "sft"
                rationale = f"NVIDIA {cuda['name']} ({vram:.0f}GB) — standard SFT."
                alternatives = ["minillm"]
            else:
                backend = "sft"
                rationale = f"Low VRAM ({vram:.0f}GB) — SFT with gradient checkpointing."
                warnings.append("Very low VRAM — set batch_size=1 and grad_accum≥8.")
                alternatives = []
        else:
            backend = "sft"
            rationale = "No GPU detected — CPU training (slow for large models)."
            warnings.append("CPU training is very slow. Recommend a smaller student model.")
            alternatives = []

    # ── User preference override ──────────────────────────────────────────
    if prefer and prefer != backend:
        available = {
            "mlx":     _has_mlx(),
            "unsloth": _has_unsloth(),
            "sft":     True,
            "minillm": True,
            "forward": True,
        }
        if available.get(prefer, False):
            backend = prefer
            rationale = f"User selected {prefer}."
        else:
            warnings.append(f"Preferred backend '{prefer}' not available — using {backend}.")

    return {
        "backend":      backend,
        "rationale":    rationale,
        "warnings":     warnings,
        "alternatives": alternatives,
    }


def all_available_backends() -> list[str]:
    """Return a list of all backends usable on this machine."""
    available = ["sft", "minillm", "forward"]
    if _has_mlx():
        available.insert(0, "mlx")
    if _has_unsloth():
        available.insert(0, "unsloth")
    return available
