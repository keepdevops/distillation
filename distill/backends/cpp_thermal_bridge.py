"""Python adapter for ThermalReading and HardwareProfile C++ structs.

Falls back gracefully to the pure-Python thermal_structured module when
the distill_cpp extension is not compiled. The UI never needs to know which
path is active — it always receives the same dict shape.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _try_cpp():
    """Return (ThermalReading, HardwareProfile) from distill_cpp, or (None, None)."""
    try:
        import distill_cpp  # type: ignore[import]
        return distill_cpp.ThermalReading, distill_cpp.HardwareProfile
    except ImportError:
        return None, None


def read_thermal_dict() -> dict[str, Any]:
    """Return a validated thermal snapshot dict, using C++ struct when available."""
    from distill.backends.struct_wrappers import validate_thermal

    ThermalReading, _ = _try_cpp()
    if ThermalReading is not None:
        try:
            # C++ path: populate from mactop via Python, store in C++ struct
            from distill.monitoring.thermal_structured import read_thermals
            snap = read_thermals()
            r = ThermalReading()
            r.cpu_temp    = snap.cpu_temp
            r.gpu_temp    = snap.gpu_temp
            r.soc_temp    = snap.soc_temp
            r.cpu_power   = snap.cpu_power
            r.gpu_power   = snap.gpu_power
            r.total_power = snap.total_power
            r.available   = snap.available
            r.error       = snap.error
            return validate_thermal(r.to_dict())
        except Exception as exc:
            logger.warning("C++ thermal path failed: %s — falling back", exc)

    # Pure-Python fallback
    try:
        from distill.monitoring.thermal_structured import read_thermals
        snap = read_thermals()
        return validate_thermal(snap.to_dict())
    except Exception as exc:
        logger.error("thermal fallback failed: %s", exc)
        return {"available": False, "error": str(exc),
                "cpu_temp": 0.0, "gpu_temp": 0.0, "soc_temp": 0.0,
                "cpu_power": 0.0, "gpu_power": 0.0, "total_power": 0.0}


def build_hardware_profile_dict() -> dict[str, Any]:
    """Return a hardware profile dict, using C++ struct when available."""
    _, HardwareProfile = _try_cpp()

    try:
        from distill.monitoring.thermal_structured import detect_hardware
        hw = detect_hardware()
    except Exception as exc:
        logger.error("detect_hardware failed: %s", exc)
        hw = {"device": "Unknown", "ram_gb": 0.0, "machine": "", "backend_hint": "cpu"}

    if HardwareProfile is not None:
        try:
            p = HardwareProfile()
            p.device_label = hw["device"]
            p.machine      = hw["machine"]
            p.backend_hint = hw["backend_hint"]
            p.ram_gb       = float(hw["ram_gb"])
            p.has_mps      = hw["machine"] == "arm64"
            return {
                "device":       p.device_label,
                "machine":      p.machine,
                "backend_hint": p.backend_hint,
                "ram_gb":       p.ram_gb,
                "has_mps":      p.has_mps,
                "has_cuda":     p.has_cuda,
                "cpp_backed":   True,
            }
        except Exception as exc:
            logger.warning("C++ HardwareProfile failed: %s — falling back", exc)

    return {**hw, "cpp_backed": False}


def oom_risk(threshold: float = 85.0) -> str:
    """Return 'low' / 'medium' / 'high' using C++ struct method when available."""
    ThermalReading, _ = _try_cpp()
    d = read_thermal_dict()
    if ThermalReading is not None:
        try:
            r = ThermalReading()
            r.cpu_temp = d["cpu_temp"]; r.gpu_temp = d["gpu_temp"]
            r.soc_temp = d["soc_temp"]; r.available = d["available"]
            return r.oom_risk(threshold)
        except Exception:
            pass
    peak = max(d.get("cpu_temp", 0.0), d.get("gpu_temp", 0.0), d.get("soc_temp", 0.0))
    if peak >= threshold:        return "high"
    if peak >= threshold * 0.85: return "medium"
    return "low"
