#!/usr/bin/env python3
"""Tests for C++ pybind11 struct bindings (distill_cpp) and Python fallback paths.

Tests both the compiled C++ path (when available) and the pure-Python
fallback to ensure both produce correct results with identical shapes.

Run with: python -m pytest tests/test_cpp_bindings.py -v
       or: python tests/test_cpp_bindings.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# ── cpp availability ──────────────────────────────────────────────────────────

def test_cpp_module_availability():
    """distill.cpp.__init__ must be importable regardless of compilation."""
    from distill.cpp import is_available, get_module
    available = is_available()
    print(f"  distill_cpp compiled: {available}")
    # Not asserting True — C++ is optional; just must not crash


# ── ThermalReading (Python fallback) ─────────────────────────────────────────

def test_thermal_bridge_dict_shape():
    from distill.backends.cpp_thermal_bridge import read_thermal_dict
    d = read_thermal_dict()
    required = {"cpu_temp", "gpu_temp", "soc_temp", "total_power", "available", "error"}
    assert required.issubset(d.keys()), f"Missing keys: {required - d.keys()}"
    assert isinstance(d["cpu_temp"], float)
    assert isinstance(d["available"], bool)


def test_thermal_bridge_hardware_profile():
    from distill.backends.cpp_thermal_bridge import build_hardware_profile_dict
    hw = build_hardware_profile_dict()
    assert "device" in hw
    assert "backend_hint" in hw
    assert isinstance(hw["ram_gb"], float)
    assert hw["ram_gb"] > 0


def test_thermal_bridge_oom_risk():
    from distill.backends.cpp_thermal_bridge import oom_risk
    risk = oom_risk(threshold=85.0)
    assert risk in ("low", "medium", "high")


# ── QuantConfig (Python fallback) ────────────────────────────────────────────

def test_quant_config_build():
    from distill.backends.llama_struct_bridge import build_quant_config, quant_preset_names
    cfg = build_quant_config("q4_k_m", output_path="/tmp/model.gguf")
    assert cfg["method"] == "q4_k_m"
    assert cfg["bits"] == 4
    assert cfg["group_size"] == 128
    assert cfg["output_path"] == "/tmp/model.gguf"
    presets = quant_preset_names()
    assert "q4_k_m" in presets
    assert "awq" in presets


def test_model_metrics_build():
    from distill.backends.llama_struct_bridge import build_model_metrics
    mm = build_model_metrics({
        "model_id": "test", "backend": "mlx",
        "tokens_per_sec": 125.0, "perplexity": 5.2,
        "quality_score": 0.85,
    })
    assert mm["tokens_per_sec"] == 125.0
    assert mm["perplexity"] == 5.2


def test_model_metrics_compare():
    from distill.backends.llama_struct_bridge import compare_metrics
    a = {"perplexity": 4.8, "quality_score": 0.90, "tokens_per_sec": 120.0}
    b = {"perplexity": 5.5, "quality_score": 0.82, "tokens_per_sec": 95.0}
    result = compare_metrics(a, b)
    assert "perplexity" in result or "overall" in result
    if "perplexity" in result:
        assert result["perplexity"] == "a"  # lower is better


# ── MetricsAdapter (Python fallback) ─────────────────────────────────────────

def test_metrics_adapter_push_series():
    from distill.backends.cpp_metrics_adapter import MetricsAdapter
    ma = MetricsAdapter()
    for i in range(10):
        ma.push(step=i * 10, loss=2.0 - i * 0.15, lr=2e-4, grad_norm=0.9)
    series = ma.get_series()
    assert len(series["steps"]) == 10
    assert len(series["loss"]) == 10
    assert len(series["smoothed"]) == 10
    assert series["steps"][-1] == 90


def test_metrics_adapter_smoothed_loss():
    from distill.backends.cpp_metrics_adapter import MetricsAdapter
    ma = MetricsAdapter()
    for i in range(20):
        ma.push(step=i, loss=float(i))
    sl = ma.smoothed_loss(window=5)
    assert sl > 0


def test_metrics_adapter_clear():
    from distill.backends.cpp_metrics_adapter import MetricsAdapter
    ma = MetricsAdapter()
    ma.push(1, 1.0)
    ma.push(2, 0.9)
    assert len(ma) == 2
    ma.clear()
    assert len(ma) == 0


# ── LoRA bridge (Python fallback) ────────────────────────────────────────────

def test_lora_config_roundtrip():
    from distill.backends.lora_bridge import build_lora_config
    cfg = build_lora_config(rank=32, alpha=64, use_qlora=True,
                             target_modules=["q_proj", "k_proj", "v_proj"])
    assert cfg["rank"] == 32
    assert cfg["alpha"] == 64
    assert cfg["use_qlora"] is True
    assert cfg["scaling"] == 2.0
    assert "q_proj" in cfg["target_modules"]


def test_lora_vram_positive():
    from distill.backends.lora_bridge import estimate_vram
    v = estimate_vram(rank=16, hidden_size=4096, num_layers=32,
                      base_model_gb=13.5, batch_size=2, seq_len=512)
    assert v["total_gb"] > 13.5
    assert v["adapter_mb"] > 0
    assert v["activations_mb"] > 0


def test_lora_training_metrics_aggregation():
    from distill.backends.lora_bridge import push_training_metrics
    m = push_training_metrics(
        step=50,
        adapter_norms=[0.12, 0.15, 0.11, 0.18],
        update_ratios=[0.004, 0.006, 0.005, 0.007],
        grad_norms=[0.9, 1.0, 0.85, 1.1],
    )
    assert m["step"] == 50
    assert m["mean_adapter_norm"] > 0
    assert m["dead_layers"] == 0
    assert m["health_ok"] is True


def test_lora_dead_layer_detection():
    from distill.backends.lora_bridge import push_training_metrics
    m = push_training_metrics(
        step=1,
        adapter_norms=[0.0, 0.0, 1e-10, 0.15],
        update_ratios=[], grad_norms=[],
    )
    assert m["dead_layers"] >= 3


# ── Export bridge (Python fallback) ──────────────────────────────────────────

def test_export_format_spec_roundtrip():
    from distill.backends.export_bridge import build_format_spec
    spec = build_format_spec("awq", bits=4, group_size=128,
                              merge_lora=True, optimize_for="speed")
    assert spec["format_key"] == "awq"
    assert spec["bits"] == 4
    assert spec["optimize_for"] == "speed"
    assert spec["merge_lora"] is True


def test_export_result_status():
    from distill.backends.export_bridge import build_result
    ok  = build_result("gguf", "/tmp/model.gguf", success=True, elapsed_sec=30.0)
    err = build_result("exl2", "", success=False, error="CUDA OOM")
    assert ok["status_icon"]  == "✅"
    assert err["status_icon"] == "❌"
    assert ok["elapsed_sec"]  == 30.0


def test_export_manifest_aggregation():
    from distill.backends.export_bridge import build_manifest, build_result
    results = [
        build_result("gguf",        "/tmp/model.gguf", success=True),
        build_result("safetensors", "/tmp/model.st",   success=True),
        build_result("awq",         "",                success=False, error="no GPU"),
    ]
    m = build_manifest("distill-v1", "/tmp/ckpt", results)
    assert m["succeeded"] == 2
    assert m["failed"]    == 1
    assert m["total_formats"] == 3


# ── Struct wrappers ───────────────────────────────────────────────────────────

def test_struct_wrappers_validate_thermal():
    from distill.backends.struct_wrappers import validate_thermal
    d = validate_thermal({"cpu_temp": "44.5", "gpu_temp": 38.0})
    assert d["cpu_temp"] == 44.5
    assert isinstance(d["cpu_temp"], float)
    assert d.get("available") is False  # defaults set


def test_struct_wrappers_validate_quant():
    from distill.backends.struct_wrappers import validate_quant_config
    d = validate_quant_config({"method": "Q4_K_M", "bits": 4})
    assert d["method"] == "q4_k_m"  # normalised to lowercase in schema
    assert d["group_size"] == 128


def test_struct_wrappers_to_dict():
    from distill.backends.struct_wrappers import to_dict
    from dataclasses import dataclass

    @dataclass
    class Dummy:
        x: int = 1
        y: float = 2.0

    d = to_dict(Dummy())
    assert d == {"x": 1, "y": 2.0}
    assert to_dict({"a": 1}) == {"a": 1}


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_cpp_module_availability,
        test_thermal_bridge_dict_shape,
        test_thermal_bridge_hardware_profile,
        test_thermal_bridge_oom_risk,
        test_quant_config_build,
        test_model_metrics_build,
        test_model_metrics_compare,
        test_metrics_adapter_push_series,
        test_metrics_adapter_smoothed_loss,
        test_metrics_adapter_clear,
        test_lora_config_roundtrip,
        test_lora_vram_positive,
        test_lora_training_metrics_aggregation,
        test_lora_dead_layer_detection,
        test_export_format_spec_roundtrip,
        test_export_result_status,
        test_export_manifest_aggregation,
        test_struct_wrappers_validate_thermal,
        test_struct_wrappers_validate_quant,
        test_struct_wrappers_to_dict,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  ❌ {t.__name__}: {exc}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(0 if failed == 0 else 1)
