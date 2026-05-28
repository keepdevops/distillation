#!/usr/bin/env python3
"""Smoke tests for Wow Sausage Maker UI components.

Tests all critical paths without launching a live Gradio server.
Run with: python -m pytest tests/test_ui_components.py -v
       or: python tests/test_ui_components.py
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# ── EventBus ──────────────────────────────────────────────────────────────────

def test_event_bus_subscribe_emit():
    from distill.ui.core.event_bus import EventBus, Topic, Event
    bus = EventBus()
    received = []
    bus.on(Topic.TRAINING_STEP, lambda e: received.append(e))
    bus.emit(Topic.TRAINING_STEP, {"step": 42, "loss": 1.23}, source="test")
    assert len(received) == 1
    assert received[0].get("step") == 42
    assert received[0].source == "test"


def test_event_bus_unsubscribe():
    from distill.ui.core.event_bus import EventBus, Topic
    bus = EventBus()
    calls = []
    h = lambda e: calls.append(e)
    bus.on(Topic.THERMAL_ALERT, h)
    bus.emit(Topic.THERMAL_ALERT, {})
    bus.off(Topic.THERMAL_ALERT, h)
    bus.emit(Topic.THERMAL_ALERT, {})
    assert len(calls) == 1


def test_event_bus_history():
    from distill.ui.core.event_bus import EventBus, Topic
    bus = EventBus()
    for i in range(5):
        bus.emit(Topic.JOB_LOG_LINE, {"line": f"log {i}"})
    recent = bus.recent(Topic.JOB_LOG_LINE, n=3)
    assert len(recent) == 3


def test_event_bus_all_topics_defined():
    from distill.ui.core.event_bus import Topic
    assert len(list(Topic)) >= 20


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_tabs():
    from distill.ui.core.registry import registry
    keys = registry.tab_keys()
    assert len(keys) >= 12
    for required in ("hardware", "swarm_export", "alignment", "training_live"):
        assert required in keys, f"Missing tab: {required}"


def test_registry_backends():
    from distill.ui.core.registry import registry
    assert registry.backend_descriptor("mlx") is not None
    assert registry.backend_descriptor("sft") is not None
    assert registry.backend_descriptor("mlx").lora_support is True
    assert registry.backend_descriptor("mlx").qlora_support is True
    assert "mlx" in registry.backend_choices(platform="mps")
    assert "mlx" not in registry.backend_choices(platform="cuda")


def test_registry_export_formats():
    from distill.ui.core.registry import registry
    assert len(registry.export_format_keys()) >= 9
    assert registry.export_descriptor("gguf").lora_merge_required is True
    assert "cuda" in registry.export_descriptor("awq").platforms
    labels = registry.export_format_choices()
    assert "GGUF (llama.cpp)" in labels
    assert registry.export_label_to_key("GGUF (llama.cpp)") == "gguf"


def test_registry_plugin_registration():
    from distill.ui.core.registry import ComponentRegistry
    r = ComponentRegistry()
    r.register_tab("custom_tab", "distill.ui.tabs.hardware", "build_tab",
                   label="Custom")
    assert "custom_tab" in r.tab_keys()
    assert r.tab_descriptor("custom_tab").label == "Custom"
    r.register_backend("custom_be", label="Custom Backend")
    assert "custom_be" in r.backend_choices()


# ── Config schemas ────────────────────────────────────────────────────────────

def test_lora_config_defaults():
    from distill.config.schemas import LoRAConfig
    lora = LoRAConfig(rank=16, alpha=32)
    assert lora.scaling == 2.0
    assert lora.estimated_params() > 0
    assert lora.estimated_vram_mb() > 0


def test_lora_config_alpha_autocorrect():
    from distill.config.schemas import LoRAConfig
    lora = LoRAConfig(rank=32, alpha=4)
    assert lora.alpha >= lora.rank


def test_training_config_validation():
    from distill.config.schemas import TrainingConfig
    import pytest
    cfg = TrainingConfig(teacher="Qwen/Qwen2-1.5B", student="Qwen/Qwen2-0.5B")
    assert cfg.effective_batch_size() == cfg.batch_size * cfg.grad_accum
    cli = cfg.to_cli_args()
    assert "--backend" in cli and "--lora_r" in cli
    try:
        TrainingConfig(teacher="x", student="y", backend="bad_backend")
        assert False, "Should have raised"
    except Exception:
        pass


def test_export_config_normalise():
    from distill.config.schemas import ExportConfig
    cfg = ExportConfig(format="GGUF", output_dir="/tmp")
    assert cfg.format == "gguf"


def test_thermal_snapshot_risk():
    from distill.config.schemas import ThermalSnapshot
    snap_ok  = ThermalSnapshot(cpu_temp=44.0, gpu_temp=39.0, available=True)
    snap_hot = ThermalSnapshot(cpu_temp=88.0, gpu_temp=91.0, available=True)
    assert snap_ok.oom_risk()  == "low"
    assert snap_hot.oom_risk() == "high"
    assert snap_hot.peak_temp() == 91.0


def test_from_preset():
    from distill.config.schemas import from_preset
    cfg = from_preset("DevOps Agent")
    assert cfg.backend == "mlx"
    assert cfg.teacher != ""


# ── LoRA bridge ───────────────────────────────────────────────────────────────

def test_lora_bridge_build_config():
    from distill.backends.lora_bridge import build_lora_config
    cfg = build_lora_config(rank=16, alpha=32)
    assert cfg["rank"] == 16
    assert cfg["scaling"] == 2.0


def test_lora_bridge_vram_estimate():
    from distill.backends.lora_bridge import estimate_vram
    result = estimate_vram(rank=32, base_model_gb=3.0, batch_size=4)
    assert result["total_gb"] > 3.0
    assert result["adapter_mb"] > 0


def test_lora_bridge_dead_layer_detection():
    from distill.backends.lora_bridge import push_training_metrics
    metrics = push_training_metrics(
        step=1, adapter_norms=[0.0, 1e-10, 0.15], update_ratios=[], grad_norms=[]
    )
    assert metrics["dead_layers"] >= 2
    assert metrics["health_ok"] is False


# ── Export bridge ─────────────────────────────────────────────────────────────

def test_export_bridge_format_spec():
    from distill.backends.export_bridge import build_format_spec
    spec = build_format_spec("gguf", quant_method="q4_k_m", merge_lora=True)
    assert spec["format_key"] == "gguf"
    assert spec["merge_lora"] is True


def test_export_bridge_result():
    from distill.backends.export_bridge import build_result
    ok  = build_result("gguf", "/tmp/x.gguf", success=True)
    err = build_result("awq", "", success=False, error="no CUDA")
    assert ok["status_icon"]  == "✅"
    assert err["status_icon"] == "❌"


def test_export_bridge_manifest():
    from distill.backends.export_bridge import build_manifest, build_result
    results = [
        build_result("gguf", "/tmp/m.gguf", success=True),
        build_result("awq",  "",            success=False, error="err"),
    ]
    m = build_manifest("my-model", "/tmp/ckpt", results)
    assert m["succeeded"] == 1
    assert m["failed"]    == 1


def test_export_bridge_labels_to_specs():
    from distill.backends.export_bridge import labels_to_specs
    specs = labels_to_specs(["GGUF (llama.cpp)", "MLX Weights"], "/tmp/out")
    assert len(specs) == 2
    assert any(s["format_key"] == "gguf" for s in specs)


# ── LoRA metrics charts ───────────────────────────────────────────────────────

def test_lora_metrics_charts():
    from distill.ui.components.lora_metrics_charts import (
        push_lora_metrics, health_summary_html, adapter_norm_figure,
        update_ratio_figure, dead_layers_figure, clear_history,
    )
    clear_history()
    for i in range(5):
        push_lora_metrics({"step": i * 10, "mean_adapter_norm": 0.1,
                           "mean_update_ratio": 0.005, "dead_layers": 0})
    assert "pill" in health_summary_html()
    assert adapter_norm_figure() is not None
    assert update_ratio_figure() is not None
    assert dead_layers_figure()  is not None


def test_lora_metrics_dead_health():
    from distill.ui.components.lora_metrics_charts import (
        push_lora_metrics, health_summary_html, clear_history
    )
    clear_history()
    push_lora_metrics({"step": 1, "mean_adapter_norm": 1e-10, "dead_layers": 3})
    html = health_summary_html()
    assert "yellow" in html or "warn" in html.lower() or "dead" in html.lower()


# ── Export matrix UI ──────────────────────────────────────────────────────────

def test_export_matrix_format_table():
    from distill.ui.components.export_matrix_ui import _format_table_html, _detect_platform
    platform = _detect_platform()
    assert platform in ("mps", "cuda", "cpu")
    html = _format_table_html(["GGUF (llama.cpp)", "AWQ (4-bit GPU)"])
    assert "GGUF" in html
    assert len(html) > 100


# ── Production pack ───────────────────────────────────────────────────────────

def test_production_pack_build():
    from distill.ui.components.production_pack import build_pack
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = build_pack(
            model_path="/tmp/my-model",
            export_results={"gguf": {"output_path": "/tmp/model.gguf", "success": True}},
            output_dir=tmp,
            system_prompt="You are helpful.",
            vllm_port=8000, llama_port=8080,
        )
        assert Path(zip_path).exists()
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert any("launch_vllm" in n for n in names)
            assert any("docker-compose" in n for n in names)
            assert any("manifest.json" in n for n in names)
            assert any("inference_example" in n for n in names)
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["model_path"] == "/tmp/my-model"
        print(f"  ZIP contains: {sorted(names)}")


# ── App build ─────────────────────────────────────────────────────────────────

def test_app_builds():
    from distill.ui.app import build_app
    app = build_app()
    assert app is not None


# ── Runner ────────────────────────────────────────────────────────────────────

import zipfile


if __name__ == "__main__":
    tests = [
        test_event_bus_subscribe_emit,
        test_event_bus_unsubscribe,
        test_event_bus_history,
        test_event_bus_all_topics_defined,
        test_registry_tabs,
        test_registry_backends,
        test_registry_export_formats,
        test_registry_plugin_registration,
        test_lora_config_defaults,
        test_lora_config_alpha_autocorrect,
        test_training_config_validation,
        test_export_config_normalise,
        test_thermal_snapshot_risk,
        test_from_preset,
        test_lora_bridge_build_config,
        test_lora_bridge_vram_estimate,
        test_lora_bridge_dead_layer_detection,
        test_export_bridge_format_spec,
        test_export_bridge_result,
        test_export_bridge_manifest,
        test_export_bridge_labels_to_specs,
        test_lora_metrics_charts,
        test_lora_metrics_dead_health,
        test_export_matrix_format_table,
        test_production_pack_build,
        test_app_builds,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  ❌ {t.__name__}: {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(0 if failed == 0 else 1)
