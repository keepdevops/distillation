"""Plugin/component registry — the single authoritative catalog of what's available.

Every tab, training backend, and export format registers itself here at import time.
The UI reads from this registry instead of hard-coding lists of options.

Usage:
    from distill.ui.core.registry import registry

    # Register (done at module level in each plugin)
    registry.register_tab("hardware", "distill.ui.tabs.hardware", "build_tab")
    registry.register_backend("mlx", label="MLX (Apple Silicon)")
    registry.register_export_format("gguf", label="GGUF (llama.cpp)", platforms=["cpu","mps","cuda"])

    # Query (done by app.py / tab selectors)
    tabs     = registry.tab_keys()
    backends = registry.backend_choices()
    formats  = registry.export_format_choices()
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Descriptor types ───────────────────────────────────────────────────────────

@dataclass
class TabDescriptor:
    key: str
    module: str
    fn: str = "build_tab"
    label: str = ""
    icon: str = ""
    requires: list[str] = field(default_factory=list)  # pip packages needed


@dataclass
class BackendDescriptor:
    key: str
    label: str
    platforms: list[str] = field(default_factory=list)  # ["mps","cuda","cpu"]
    requires: list[str] = field(default_factory=list)
    lora_support: bool = True
    qlora_support: bool = False
    description: str = ""


@dataclass
class ExportFormatDescriptor:
    key: str
    label: str
    platforms: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    lora_merge_required: bool = False
    description: str = ""


# ── Registry ───────────────────────────────────────────────────────────────────

class ComponentRegistry:
    """Central registry for all pluggable components."""

    def __init__(self) -> None:
        self._tabs:    dict[str, TabDescriptor]          = {}
        self._backends: dict[str, BackendDescriptor]     = {}
        self._formats:  dict[str, ExportFormatDescriptor] = {}
        self._builder_cache: dict[str, Callable] = {}

    # ── Tab registration ───────────────────────────────────────────────────

    def register_tab(
        self, key: str, module: str, fn: str = "build_tab",
        label: str = "", icon: str = "", requires: list[str] | None = None,
    ) -> None:
        self._tabs[key] = TabDescriptor(
            key=key, module=module, fn=fn,
            label=label or key.replace("_", " ").title(),
            icon=icon, requires=requires or [],
        )
        self._builder_cache.pop(key, None)

    def build_tab(self, key: str) -> None:
        """Invoke the tab builder for key (lazy import, cached)."""
        import gradio as gr
        builder = self._resolve_builder(key)
        if builder is None:
            gr.Markdown(f"⚠ Tab `{key}` not registered or failed to load.")
            return
        try:
            builder()
        except Exception as exc:
            logger.error("Tab '%s' failed: %s", key, exc, exc_info=True)
            gr.Markdown(f"**Tab `{key}` error:** `{exc}`")

    def tab_keys(self) -> list[str]:
        return list(self._tabs.keys())

    def tab_descriptor(self, key: str) -> TabDescriptor | None:
        return self._tabs.get(key)

    def _resolve_builder(self, key: str) -> Callable | None:
        if key in self._builder_cache:
            return self._builder_cache[key]
        desc = self._tabs.get(key)
        if not desc:
            logger.warning("Unknown tab key: '%s'", key)
            return None
        try:
            mod = importlib.import_module(desc.module)
            fn  = getattr(mod, desc.fn)
            self._builder_cache[key] = fn
            return fn
        except (ImportError, AttributeError) as exc:
            logger.warning("Tab '%s' load failed: %s", key, exc)
            return None

    # ── Backend registration ───────────────────────────────────────────────

    def register_backend(
        self, key: str, label: str = "", platforms: list[str] | None = None,
        requires: list[str] | None = None, lora_support: bool = True,
        qlora_support: bool = False, description: str = "",
    ) -> None:
        self._backends[key] = BackendDescriptor(
            key=key, label=label or key.upper(),
            platforms=platforms or [], requires=requires or [],
            lora_support=lora_support, qlora_support=qlora_support,
            description=description,
        )

    def backend_choices(self, platform: str | None = None) -> list[str]:
        """Return backend keys, optionally filtered by platform."""
        keys = list(self._backends.keys())
        if platform:
            keys = [k for k in keys
                    if not self._backends[k].platforms
                    or platform in self._backends[k].platforms]
        return keys

    def backend_dropdown_choices(self) -> list[tuple[str, str]]:
        """Return (label, key) pairs for gr.Dropdown."""
        return [(d.label, d.key) for d in self._backends.values()]

    def backend_descriptor(self, key: str) -> BackendDescriptor | None:
        return self._backends.get(key)

    # ── Export format registration ─────────────────────────────────────────

    def register_export_format(
        self, key: str, label: str = "", platforms: list[str] | None = None,
        requires: list[str] | None = None, lora_merge_required: bool = False,
        description: str = "",
    ) -> None:
        self._formats[key] = ExportFormatDescriptor(
            key=key, label=label or key.upper(),
            platforms=platforms or [], requires=requires or [],
            lora_merge_required=lora_merge_required, description=description,
        )

    def export_format_keys(self) -> list[str]:
        return list(self._formats.keys())

    def export_format_choices(self) -> list[str]:
        """Return UI labels for gr.CheckboxGroup."""
        return [d.label for d in self._formats.values()]

    def export_label_to_key(self, label: str) -> str | None:
        for d in self._formats.values():
            if d.label == label:
                return d.key
        return None

    def export_descriptor(self, key: str) -> ExportFormatDescriptor | None:
        return self._formats.get(key)

    # ── Introspection ──────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        return {
            "tabs":    len(self._tabs),
            "backends": len(self._backends),
            "export_formats": len(self._formats),
            "tab_keys": self.tab_keys(),
            "backend_keys": self.backend_choices(),
            "format_keys": self.export_format_keys(),
        }


# ── Module-level singleton ─────────────────────────────────────────────────────

registry = ComponentRegistry()


def _register_defaults() -> None:
    """Register all built-in tabs, backends, and export formats."""

    # ── Tabs ──────────────────────────────────────────────────────────────
    _tabs = [
        ("hardware",         "distill.ui.tabs.hardware",         "build_tab", "⚙ Hardware",        "⚙"),
        ("data_prep",        "distill.launch_ui.tabs.tab_data_prep", "build_tab_data_prep", "📦 Data", "📦"),
        ("configure_backend","distill.ui.tabs.configure_backend", "build_tab", "🎛 Configure",      "🎛"),
        ("sft",              "distill.launch_ui.tabs.tab_configure", "build_tab_configure", "🎓 SFT", "🎓"),
        ("distillation",     "distill.launch_ui.tabs.tab_expert",  "build_tab_expert", "🧠 Distil", "🧠"),
        ("alignment",        "distill.ui.tabs.alignment",          "build_tab", "⚖ Align",          "⚖"),
        ("training_live",    "distill.ui.tabs.training_live",      "build_tab", "📈 Live",           "📈"),
        ("eval_comparison",  "distill.ui.tabs.eval_comparison",    "build_tab", "📊 Eval",           "📊"),
        ("quantize_export",  "distill.ui.tabs.quantize_export",    "build_tab", "📤 Export",         "📤"),
        ("swarm_export",     "distill.ui.tabs.swarm_export",       "build_tab", "🚀 Swarm",          "🚀"),
        ("full_auto_gantt",  "distill.ui.tabs.full_auto_gantt",    "build_tab", "▶ Full Auto",      "▶"),
        ("experiments",      "distill.ui.tabs.experiments",        "build_tab", "🔬 Experiments",    "🔬"),
        ("logs",             "distill.launch_ui.tabs.tab_logs",    "build_tab_logs", "📋 Logs",      "📋"),
        ("help",             "distill.launch_ui.tabs.tab_help",    "build_tab_help", "❓ Help",      "❓"),
    ]
    for key, module, fn, label, icon in _tabs:
        registry.register_tab(key, module, fn, label=label, icon=icon)

    # ── Backends ──────────────────────────────────────────────────────────
    _backends = [
        ("mlx",     "MLX (Apple Silicon)",     ["mps"],         [],            True,  True,  "Native Apple Silicon. Fastest on M-series."),
        ("sft",     "SFT (HF Trainer)",        ["mps","cuda","cpu"], [],       True,  False, "Standard supervised fine-tuning via HF Trainer."),
        ("minillm", "MiniLLM Distillation",    ["mps","cuda","cpu"], [],       True,  False, "Reverse-KL distillation from teacher logits."),
        ("unsloth", "Unsloth (Fast QLoRA)",    ["cuda"],        ["unsloth"],   True,  True,  "2-5× faster LoRA/QLoRA on NVIDIA GPUs."),
        ("forward", "Forward KD",              ["mps","cuda","cpu"], [],       True,  False, "Forward KL divergence distillation."),
    ]
    for key, label, platforms, reqs, lora, qlora, desc in _backends:
        registry.register_backend(key, label=label, platforms=platforms,
                                   requires=reqs, lora_support=lora,
                                   qlora_support=qlora, description=desc)

    # ── Export formats ─────────────────────────────────────────────────────
    _formats = [
        ("gguf",        "GGUF (llama.cpp)",     ["cpu","mps","cuda"], [],              True,  "Universal GGUF for llama.cpp inference."),
        ("mlx",         "MLX Weights",          ["mps"],              [],              False, "Native MLX format for Apple Silicon."),
        ("coreml",      "CoreML",               ["mps"],              [],              True,  "On-device macOS/iOS inference."),
        ("safetensors", "Safetensors + HF Hub", ["cpu","mps","cuda"], ["huggingface-hub"], True, "Standard HF format, push to Hub."),
        ("awq",         "AWQ (4-bit GPU)",      ["cuda"],             ["autoawq"],     True,  "Activation-aware weight quantization."),
        ("gptq",        "GPTQ (4-bit GPU)",     ["cuda"],             ["auto-gptq"],   True,  "Classic 4-bit GPU quantization."),
        ("exl2",        "EXL2 (high-perf GPU)", ["cuda"],             ["exllamav2"],   True,  "Mixed-precision, best single-GPU quality."),
        ("vllm_config", "vLLM Config",          ["cuda"],             [],              False, "vLLM server config + launch script."),
        ("onnx",        "ONNX",                 ["cpu","mps","cuda"], ["optimum"],     True,  "Cross-platform / edge deployment."),
    ]
    for key, label, platforms, reqs, merge, desc in _formats:
        registry.register_export_format(key, label=label, platforms=platforms,
                                         requires=reqs, lora_merge_required=merge,
                                         description=desc)


_register_defaults()
