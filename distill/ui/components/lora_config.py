"""Reusable LoRA / QLoRA configuration panel with live VRAM estimation.

Drop this into any training tab with:
    from distill.ui.components.lora_config import LoRAPanel
    panel = LoRAPanel()
    widgets = panel.render()
    # Later: config_dict = panel.to_config()

The panel auto-updates VRAM estimates when rank, batch size, or model size changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

# ── Model size lookup (hidden_size, num_layers) for common students ───────────
_MODEL_DIMS: dict[str, tuple[int, int]] = {
    "0.5b": (896,  24),
    "1b":   (2048, 24),
    "1.5b": (1536, 28),
    "3b":   (3072, 28),
    "7b":   (4096, 32),
    "8b":   (4096, 32),
    "13b":  (5120, 40),
}

_DEFAULT_BASE_VRAM = {
    "mlx":     {"0.5b": 1.2, "1b": 2.1, "1.5b": 3.0, "3b": 5.8, "7b": 13.5},
    "sft":     {"0.5b": 1.5, "1b": 2.8, "1.5b": 3.8, "3b": 7.0, "7b": 15.0},
    "unsloth": {"0.5b": 0.8, "1b": 1.4, "1.5b": 2.0, "3b": 3.5, "7b": 7.0},
}


def _model_size_key(student: str) -> str:
    lower = student.lower()
    for key in sorted(_MODEL_DIMS.keys(), reverse=True):
        if key in lower:
            return key
    return "1.5b"


def _estimate_vram(rank: int, alpha: int, batch: int, seq_len: int,
                   backend: str, student: str, use_qlora: bool) -> str:
    """Return a human-readable VRAM estimate string."""
    try:
        from distill.backends.lora_bridge import estimate_vram
        size_key = _model_size_key(student)
        hidden, layers = _MODEL_DIMS.get(size_key, (2048, 24))
        base_vram = _DEFAULT_BASE_VRAM.get(backend, {}).get(size_key, 3.0)
        if use_qlora:
            base_vram *= 0.5

        result = estimate_vram(
            rank=rank, hidden_size=hidden, num_layers=layers,
            num_targets=2, base_model_gb=base_vram,
            batch_size=batch, seq_len=seq_len,
        )
        total = result["total_gb"]
        color = "#22c55e" if total < 8 else ("#f59e0b" if total < 16 else "#ef4444")
        return (
            f'<div style="font-size:.85rem;margin-top:.3rem">'
            f'  Estimated VRAM: <b style="color:{color}">{total:.1f} GB</b>'
            f'  <span style="color:#475569;font-size:.75rem"> '
            f'    (base {result["base_model_gb"]:.1f}GB + adapters {result["adapter_mb"]:.0f}MB'
            f'    + activations {result["activations_mb"]:.0f}MB)</span>'
            f'</div>'
        )
    except Exception as exc:
        logger.debug("VRAM estimate error: %s", exc)
        return "<small style='color:#475569'>VRAM estimate unavailable</small>"


@dataclass
class LoRAWidgets:
    rank:    Any
    alpha:   Any
    dropout: Any
    use_qlora: Any
    qlora_bits: Any
    targets:   Any
    vram_html: Any


class LoRAPanel:
    """Encapsulates the LoRA/QLoRA configuration Gradio panel."""

    def __init__(self, show_advanced: bool = True) -> None:
        self._show_advanced = show_advanced
        self._widgets: LoRAWidgets | None = None
        self._backend = "mlx"
        self._student = "Qwen/Qwen2-0.5B-Instruct"
        self._batch   = 4
        self._seq_len = 512

    def render(self, backend: str = "mlx", student: str = "") -> LoRAWidgets:
        """Render the panel inside the current gr.Blocks context."""
        self._backend = backend
        self._student = student or self._student

        with gr.Group():
            gr.Markdown(
                '<div class="section-header">LoRA / QLoRA Configuration</div>'
            )
            with gr.Row():
                rank = gr.Slider(4, 256, value=16, step=4, label="Rank (r)",
                                 info="Larger = more capacity, more VRAM")
                alpha = gr.Slider(4, 512, value=32, step=4, label="Alpha (α)",
                                  info="Scaling = α/r. Keep at 2×rank for stability")
                dropout = gr.Slider(0.0, 0.3, value=0.05, step=0.01,
                                    label="Dropout", info="0 is common for small models")

            with gr.Row():
                use_qlora = gr.Checkbox(label="QLoRA (4-bit base model)", value=False,
                                        info="Halves base VRAM at slight quality cost")
                qlora_bits = gr.Dropdown(choices=["4", "8"], value="4",
                                          label="QLoRA bits", visible=False)
                targets = gr.Dropdown(
                    choices=["q_proj,v_proj", "q_proj,k_proj,v_proj,o_proj",
                             "all-linear", "q_proj,v_proj,gate_proj,up_proj,down_proj"],
                    value="q_proj,v_proj",
                    label="Target modules",
                    allow_custom_value=True,
                )
            use_qlora.change(fn=lambda v: gr.update(visible=v),
                             inputs=use_qlora, outputs=qlora_bits)

            vram_html = gr.HTML(
                value=lambda: _estimate_vram(16, 32, self._batch, self._seq_len,
                                             self._backend, self._student, False)
            )

            def update_vram(r, a, b, qlora):
                return _estimate_vram(r, a, self._batch, self._seq_len,
                                      self._backend, self._student, qlora)

            for widget in (rank, alpha, use_qlora):
                widget.change(fn=update_vram, inputs=[rank, alpha, use_qlora, use_qlora],
                              outputs=vram_html)

        self._widgets = LoRAWidgets(
            rank=rank, alpha=alpha, dropout=dropout,
            use_qlora=use_qlora, qlora_bits=qlora_bits,
            targets=targets, vram_html=vram_html,
        )
        return self._widgets

    def to_config(self) -> dict[str, Any]:
        """Read current widget values and return a LoRAConfig dict."""
        if self._widgets is None:
            return {}
        from distill.backends.lora_bridge import build_lora_config
        rank    = int(self._widgets.rank.value)
        alpha   = int(self._widgets.alpha.value)
        dropout = float(self._widgets.dropout.value)
        qlora   = bool(self._widgets.use_qlora.value)
        targets = [t.strip() for t in self._widgets.targets.value.split(",")]
        return build_lora_config(rank=rank, alpha=alpha, dropout=dropout,
                                  use_qlora=qlora, target_modules=targets)

    def set_context(self, backend: str = "", student: str = "",
                    batch_size: int = 0, seq_len: int = 0) -> None:
        """Update context for VRAM estimation (call when config changes)."""
        if backend:   self._backend = backend
        if student:   self._student = student
        if batch_size: self._batch  = batch_size
        if seq_len:   self._seq_len = seq_len
