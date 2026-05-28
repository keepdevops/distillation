"""Reusable hardware gauge components (CPU%, RAM, GPU temp, power).

Renders as HTML metric cards that can be placed in any tab.
Separate from hardware.py to allow reuse in training_live.py sidebar.
"""
from __future__ import annotations

import logging
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)


def _color_for(value: float, warn: float, crit: float) -> str:
    if value >= crit:
        return "#ef4444"
    if value >= warn:
        return "#f59e0b"
    return "#22c55e"


def _bar_html(pct: float, color: str) -> str:
    """Return a narrow progress bar HTML snippet."""
    pct = max(0.0, min(100.0, pct))
    return (
        f'<div style="background:#1e1e2e;border-radius:2px;height:4px;margin-top:4px">'
        f'  <div style="width:{pct:.0f}%;background:{color};height:4px;border-radius:2px"></div>'
        f'</div>'
    )


def _gauge_card(label: str, value_str: str, sub: str, color: str, pct: float | None = None) -> str:
    bar = _bar_html(pct, color) if pct is not None else ""
    return (
        f'<div class="metric-card" style="min-width:120px">'
        f'  <div class="metric-value" style="color:{color};font-size:1.4rem">{value_str}</div>'
        f'  <div class="metric-label">{label}</div>'
        f'  <div style="color:#94a3b8;font-size:0.7rem;margin-top:2px">{sub}</div>'
        f'  {bar}'
        f'</div>'
    )


def build_gauges_html() -> str:
    """Build a full row of hardware gauges from live readings."""
    import psutil

    cards: list[str] = []

    # CPU usage
    try:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        color = _color_for(cpu_pct, 70, 90)
        cards.append(_gauge_card("CPU", f"{cpu_pct:.0f}%", "utilisation", color, cpu_pct))
    except Exception:
        cards.append(_gauge_card("CPU", "N/A", "", "#94a3b8"))

    # RAM
    try:
        ram = psutil.virtual_memory()
        used_gb = ram.used / (1024 ** 3)
        total_gb = ram.total / (1024 ** 3)
        pct = ram.percent
        color = _color_for(pct, 70, 90)
        cards.append(_gauge_card("RAM", f"{used_gb:.1f}GB", f"/ {total_gb:.0f}GB", color, pct))
    except Exception:
        cards.append(_gauge_card("RAM", "N/A", "", "#94a3b8"))

    # Disk
    try:
        disk = psutil.disk_usage("/")
        pct = disk.percent
        free_gb = disk.free / (1024 ** 3)
        color = _color_for(pct, 80, 95)
        cards.append(_gauge_card("Disk", f"{free_gb:.0f}GB", "free", color, pct))
    except Exception:
        cards.append(_gauge_card("Disk", "N/A", "", "#94a3b8"))

    # Thermal (from mactop)
    try:
        from distill.monitoring.thermal_structured import read_thermals
        snap = read_thermals()
        if snap.available:
            peak = max(snap.cpu_temp, snap.gpu_temp)
            color = _color_for(peak, 72, 85)
            cards.append(_gauge_card(
                "Peak Temp", f"{peak:.0f}°C",
                f"CPU {snap.cpu_temp:.0f} / GPU {snap.gpu_temp:.0f}",
                color, (peak / 100) * 100,
            ))
            cards.append(_gauge_card(
                "Power", f"{snap.total_power:.0f}W", "total draw",
                "#6366f1",
            ))
    except Exception:
        pass

    row = (
        '<div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin-bottom:0.75rem">'
        + "".join(cards)
        + "</div>"
    )
    return row


def render_gauge_row(every: int = 5) -> gr.HTML:
    """Render a live-updating gauge row inside the current gr.Blocks context."""
    return gr.HTML(value=build_gauges_html, every=every, label="")
