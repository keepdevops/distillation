"""Structured thermal reading wrapper using mactop (Apple Silicon M-series).

Wraps the existing ThermalAgent mactop call into a clean dataclass that the
UI and C++ bridge can consume without pulling in the full agent machinery.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MACTOP_BIN = "/opt/homebrew/bin/mactop"


@dataclass
class ThermalSnapshot:
    cpu_temp: float = 0.0
    gpu_temp: float = 0.0
    soc_temp: float = 0.0
    cpu_power: float = 0.0
    gpu_power: float = 0.0
    total_power: float = 0.0
    available: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "cpu_temp": self.cpu_temp,
            "gpu_temp": self.gpu_temp,
            "soc_temp": self.soc_temp,
            "cpu_power": self.cpu_power,
            "gpu_power": self.gpu_power,
            "total_power": self.total_power,
            "available": self.available,
            "error": self.error,
        }

    def oom_risk(self, threshold: float = 85.0) -> str:
        """Return 'low' / 'medium' / 'high' based on peak temp vs threshold."""
        peak = max(self.cpu_temp, self.gpu_temp, self.soc_temp)
        if peak >= threshold:
            return "high"
        if peak >= threshold * 0.85:
            return "medium"
        return "low"

    def status_pill(self, threshold: float = 85.0) -> str:
        """Return an HTML pill badge reflecting thermal status."""
        risk = self.oom_risk(threshold)
        mapping = {
            "low": ("pill-green", "● Thermal OK"),
            "medium": ("pill-yellow", "▲ Warm"),
            "high": ("pill-red", "■ Hot"),
        }
        css, label = mapping[risk]
        return f'<span class="pill {css}">{label}</span>'


def read_thermals() -> ThermalSnapshot:
    """Query mactop once and return a ThermalSnapshot.

    Falls back gracefully if mactop is not installed or returns no data.
    """
    try:
        result = subprocess.run(
            [MACTOP_BIN, "--headless", "--format", "json", "--count", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        data = json.loads(result.stdout)
        if not data:
            raise ValueError("mactop returned empty JSON")

        soc = data[0].get("soc_metrics", {})
        return ThermalSnapshot(
            cpu_temp=float(soc.get("cpu_temp", 0.0)),
            gpu_temp=float(soc.get("gpu_temp", 0.0)),
            soc_temp=float(soc.get("soc_temp", 0.0)),
            cpu_power=float(soc.get("cpu_power", 0.0)),
            gpu_power=float(soc.get("gpu_power", 0.0)),
            total_power=float(soc.get("total_power", 0.0)),
            available=True,
        )

    except FileNotFoundError:
        logger.warning("mactop not found at %s", MACTOP_BIN)
        return ThermalSnapshot(error="mactop not installed")
    except subprocess.TimeoutExpired:
        logger.warning("mactop timed out")
        return ThermalSnapshot(error="mactop timeout")
    except (json.JSONDecodeError, KeyError, ValueError, RuntimeError) as exc:
        logger.error("thermal read failed: %s", exc)
        return ThermalSnapshot(error=str(exc))


def detect_hardware() -> dict:
    """Return basic hardware profile dict (device type, RAM, backend hint)."""
    import platform
    import psutil

    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    machine = platform.machine()
    node = platform.node()

    backend_hint = "cpu"
    device_label = "Unknown"

    if machine == "arm64":
        device_label = f"Apple Silicon ({platform.processor() or 'M-series'})"
        backend_hint = "mlx" if ram_gb >= 16 else "mlx"  # always mlx on AS
    else:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                device_label = f"{gpu_name} ({vram_gb:.0f}GB VRAM)"
                backend_hint = "unsloth" if vram_gb >= 16 else "sft"
        except Exception:
            pass

    return {
        "device": device_label,
        "ram_gb": round(ram_gb, 1),
        "machine": machine,
        "node": node,
        "backend_hint": backend_hint,
    }
