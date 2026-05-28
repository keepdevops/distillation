"""Hardware tab — live thermal gauges, device profile, and backend recommendation.

Calls distill.monitoring.thermal_structured for all data; no direct mactop
invocation here. Charts live in hardware_charts.py.
"""
from __future__ import annotations

import logging

import gradio as gr

logger = logging.getLogger(__name__)


def _hw_info_html() -> str:
    """Return an HTML card with static hardware profile info."""
    try:
        from distill.monitoring.thermal_structured import detect_hardware
        hw = detect_hardware()
        risk_color = "#22c55e"
        return (
            f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));'
            f'gap:0.75rem;margin-bottom:1rem">'
            f'  <div class="metric-card"><div class="metric-value" style="color:{risk_color}">'
            f'    {hw["device"]}</div>'
            f'    <div class="metric-label">Device</div></div>'
            f'  <div class="metric-card"><div class="metric-value">{hw["ram_gb"]:.0f} GB</div>'
            f'    <div class="metric-label">System RAM</div></div>'
            f'  <div class="metric-card"><div class="metric-value" style="color:#6366f1">'
            f'    {hw["backend_hint"].upper()}</div>'
            f'    <div class="metric-label">Recommended Backend</div></div>'
            f'</div>'
        )
    except Exception as exc:
        logger.warning("hw_info_html failed: %s", exc)
        return "<p>Hardware detection unavailable.</p>"


def _thermal_html() -> str:
    """Return an HTML card with live thermal readings."""
    try:
        from distill.monitoring.thermal_structured import read_thermals
        s = read_thermals()
        if not s.available:
            return f'<div class="banner-warning">⚠ Thermal data unavailable: {s.error}</div>'

        risk = s.oom_risk()
        color_map = {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444"}
        c = color_map.get(risk, "#94a3b8")

        return (
            f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));'
            f'gap:0.75rem">'
            f'  <div class="metric-card"><div class="metric-value" style="color:{c}">'
            f'    {s.cpu_temp:.1f}°C</div><div class="metric-label">CPU Temp</div></div>'
            f'  <div class="metric-card"><div class="metric-value" style="color:{c}">'
            f'    {s.gpu_temp:.1f}°C</div><div class="metric-label">GPU Temp</div></div>'
            f'  <div class="metric-card"><div class="metric-value" style="color:{c}">'
            f'    {s.soc_temp:.1f}°C</div><div class="metric-label">SoC Temp</div></div>'
            f'  <div class="metric-card"><div class="metric-value">'
            f'    {s.total_power:.1f} W</div><div class="metric-label">Total Power</div></div>'
            f'  <div class="metric-card"><div class="metric-value" style="font-size:1rem">'
            f'    {s.status_pill()}</div><div class="metric-label">OOM Risk</div></div>'
            f'</div>'
        )
    except Exception as exc:
        logger.warning("thermal_html failed: %s", exc)
        return "<p>Thermal data unavailable.</p>"


def build_tab() -> None:
    """Render the Hardware tab inside the current gr.Blocks context."""
    gr.Markdown("## ⚙ Hardware Monitor")
    gr.Markdown(
        "Live thermal readings and hardware profile. "
        "The backend recommendation is auto-selected based on available compute."
    )

    # ── Device profile ────────────────────────────────────────────────────
    gr.Markdown("### Device Profile")
    hw_card = gr.HTML(value=_hw_info_html)

    # ── Live thermal gauges ───────────────────────────────────────────────
    gr.Markdown("### Live Thermal Status")
    thermal_card = gr.HTML(value=_thermal_html, every=10)

    # ── Thermal time-series chart ─────────────────────────────────────────
    try:
        from distill.ui.tabs.hardware_charts import render_thermal_chart
        render_thermal_chart()
    except ImportError:
        gr.Markdown("*Install plotly for thermal time-series charts.*")

    # ── Controls ──────────────────────────────────────────────────────────
    with gr.Row():
        refresh_btn = gr.Button("↻ Refresh Now", variant="secondary", scale=0)
        refresh_btn.click(fn=_hw_info_html,   outputs=hw_card)
        refresh_btn.click(fn=_thermal_html,   outputs=thermal_card)

    # ── Recommended config preview ────────────────────────────────────────
    gr.Markdown("### Recommended Configuration")
    _render_recommended_config()


def _render_recommended_config() -> None:
    """Show a config preset based on detected hardware."""
    try:
        from distill.monitoring.thermal_structured import detect_hardware
        hw = detect_hardware()
        backend = hw["backend_hint"]
        ram = hw["ram_gb"]

        if backend == "mlx":
            rec = (
                f"**Backend:** MLX (Apple Silicon detected, {ram:.0f} GB unified memory)\n\n"
                "| Parameter | Recommended |\n|---|---|\n"
                "| batch_size | 4–8 |\n"
                "| lora_rank | 16 |\n"
                "| grad_accum | 4 |\n"
                "| fp_precision | bfloat16 |"
            )
        else:
            rec = (
                f"**Backend:** {backend.upper()}\n\n"
                "Adjust batch size based on VRAM. See the SFT tab for full config."
            )
        gr.Markdown(rec)
    except Exception as exc:
        gr.Markdown(f"*Could not generate recommendation: {exc}*")
