"""Visual export format selector UI component.

Reusable across the Export tab, Swarm Export tab, and Configure Backend tab.
Renders a CheckboxGroup with per-format compatibility badges and a live
compatibility warning when formats are selected that need unavailable hardware.
"""
from __future__ import annotations

import logging
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

# Platform badge colours
_PLATFORM_COLORS = {
    "cpu":  ("#94a3b8", "CPU"),
    "mps":  ("#a78bfa", "Apple Silicon"),
    "cuda": ("#34d399", "NVIDIA GPU"),
}


def _detect_platform() -> str:
    """Return 'mps', 'cuda', or 'cpu' for the current machine."""
    try:
        import torch
        if torch.backends.mps.is_available():  return "mps"
        if torch.cuda.is_available():           return "cuda"
    except ImportError:
        import platform
        if platform.machine() == "arm64":
            return "mps"
    return "cpu"


def _format_table_html(selected: list[str]) -> str:
    """Return an HTML compatibility table for the selected formats."""
    from distill.ui.core.registry import registry
    platform = _detect_platform()

    if not selected:
        return "<small style='color:#475569'>Select formats above to see compatibility.</small>"

    rows = []
    warnings = []
    for label in selected:
        key = registry.export_label_to_key(label)
        if not key:
            continue
        desc = registry.export_descriptor(key)
        if not desc:
            continue
        platforms = desc.platforms or ["cpu", "mps", "cuda"]
        compatible = platform in platforms or not platforms
        status = "✅" if compatible else "⚠️"
        if not compatible:
            warnings.append(f"{label} requires {'/'.join(platforms)}, you have {platform.upper()}")
        badges = " ".join(
            f'<span class="pill pill-{"blue" if p == platform else "gray"}">{p}</span>'
            for p in platforms
        )
        merge_note = "🔀 merge LoRA" if desc.lora_merge_required else ""
        rows.append(
            f"<tr><td>{status} {label}</td><td>{badges}</td>"
            f"<td style='color:#94a3b8;font-size:.75rem'>{merge_note}</td></tr>"
        )

    table = (
        "<table style='font-size:.8rem;width:100%'>"
        "<tr><th>Format</th><th>Platforms</th><th>Notes</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    warn_html = ""
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        warn_html = (
            f'<div class="banner-warning">⚠ Compatibility issues:<ul '
            f'style="margin:.25rem 0 0 1rem;padding:0">{items}</ul></div>'
        )
    return warn_html + table


class ExportMatrixUI:
    """Gradio export format selector with live compatibility checking."""

    def __init__(self, default_formats: list[str] | None = None) -> None:
        from distill.ui.core.registry import registry
        self._all_labels = registry.export_format_choices()
        self._defaults = default_formats or [
            "GGUF (llama.cpp)", "MLX Weights", "Safetensors + HF Hub"
        ]
        self._checkbox: Any = None
        self._compat_html: Any = None

    def render(self, compact: bool = False) -> tuple[Any, Any]:
        """Render the matrix inside the current gr.Blocks context.

        Returns (checkbox_group, compat_html).
        """
        if compact:
            self._checkbox = gr.CheckboxGroup(
                choices=self._all_labels,
                value=self._defaults,
                label="Export Formats",
            )
            self._compat_html = gr.HTML(
                value=lambda: _format_table_html(self._defaults)
            )
        else:
            gr.Markdown("#### Export Format Matrix")
            self._checkbox = gr.CheckboxGroup(
                choices=self._all_labels,
                value=self._defaults,
                label="Target Formats",
                info="Select all formats to include in this export run.",
            )
            self._compat_html = gr.HTML(
                value=lambda: _format_table_html(self._defaults)
            )

        self._checkbox.change(
            fn=_format_table_html,
            inputs=self._checkbox,
            outputs=self._compat_html,
        )
        return self._checkbox, self._compat_html

    def selected_specs(
        self,
        output_dir: str = "exports",
        quant_method: str = "q4_k_m",
        merge_lora: bool = True,
    ) -> list[dict[str, Any]]:
        """Return ExportFormatSpec dicts for currently checked formats."""
        from distill.backends.export_bridge import labels_to_specs
        if self._checkbox is None:
            return []
        labels = self._checkbox.value or self._defaults
        return labels_to_specs(labels, output_dir, quant_method, merge_lora)

    @property
    def checkbox(self) -> Any:
        return self._checkbox

    @property
    def compat_html(self) -> Any:
        return self._compat_html
