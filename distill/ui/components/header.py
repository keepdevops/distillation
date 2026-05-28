"""Global app header with hardware status pills.

Renders a slim banner at the top of every page showing:
  • App title + version
  • Live thermal status pill (green / yellow / red)
  • Active backend hint
  • RAM available
"""
from __future__ import annotations

import logging

import gradio as gr

logger = logging.getLogger(__name__)

_APP_TITLE = "🌭 Wow Sausage Maker"
_SUBTITLE = "LLM Distillation Control Center"


def _snapshot_html() -> str:
    """Build the header HTML from a fresh thermal + hardware snapshot."""
    try:
        from distill.monitoring.thermal_structured import read_thermals, detect_hardware
        snap = read_thermals()
        hw = detect_hardware()

        pill = snap.status_pill() if snap.available else (
            '<span class="pill pill-gray">○ Thermal N/A</span>'
        )
        temp_str = (
            f"CPU {snap.cpu_temp:.0f}°C · GPU {snap.gpu_temp:.0f}°C"
            if snap.available else "–"
        )
        backend_pill = (
            f'<span class="pill pill-blue">⚡ {hw["backend_hint"].upper()}</span>'
        )
        ram_str = f'{hw["ram_gb"]:.0f} GB RAM'

        return (
            f'<div style="display:flex;align-items:center;gap:1rem;'
            f'padding:0.6rem 1rem;border-bottom:1px solid #1e293b;'
            f'background:#0f0f1a;flex-wrap:wrap;">'
            f'  <div style="font-weight:800;font-size:1.1rem;color:#e2e8f0">'
            f'    {_APP_TITLE}'
            f'  </div>'
            f'  <div style="color:#475569;font-size:0.8rem">{_SUBTITLE}</div>'
            f'  <div style="margin-left:auto;display:flex;gap:0.5rem;align-items:center;'
            f'       flex-wrap:wrap;">'
            f'    {pill}'
            f'    <span style="color:#94a3b8;font-size:0.75rem">{temp_str}</span>'
            f'    {backend_pill}'
            f'    <span class="pill pill-gray">💾 {ram_str}</span>'
            f'  </div>'
            f'</div>'
        )
    except Exception as exc:
        logger.warning("Header snapshot failed: %s", exc)
        return (
            f'<div style="padding:0.6rem 1rem;background:#0f0f1a;'
            f'border-bottom:1px solid #1e293b;font-weight:800;color:#e2e8f0">'
            f'{_APP_TITLE} — {_SUBTITLE}</div>'
        )


def render_header() -> None:
    """Render the header HTML block inside the current gr.Blocks context."""
    header_html = gr.HTML(value=_snapshot_html, every=30)  # refresh every 30s

    # Provide a manual refresh button for the hardware status
    with gr.Row():
        refresh_btn = gr.Button("↻ Refresh Status", size="sm", variant="secondary",
                                scale=0)
        refresh_btn.click(fn=_snapshot_html, outputs=header_html)
