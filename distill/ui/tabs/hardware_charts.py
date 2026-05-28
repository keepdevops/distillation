"""Plotly-based thermal time-series charts for the Hardware tab.

Separated from hardware.py to keep both files under 300 LOC and because
Plotly is an optional dependency.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from threading import Lock
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

# ── Rolling thermal history (shared across refresh calls) ─────────────────────
_MAX_POINTS = 120
_lock = Lock()
_history: dict[str, deque] = {
    "ts":         deque(maxlen=_MAX_POINTS),
    "cpu_temp":   deque(maxlen=_MAX_POINTS),
    "gpu_temp":   deque(maxlen=_MAX_POINTS),
    "soc_temp":   deque(maxlen=_MAX_POINTS),
    "total_power": deque(maxlen=_MAX_POINTS),
}


def _push_reading() -> None:
    """Append the latest thermal snapshot to the rolling history."""
    try:
        from distill.monitoring.thermal_structured import read_thermals
        s = read_thermals()
        if not s.available:
            return
        with _lock:
            _history["ts"].append(time.time())
            _history["cpu_temp"].append(s.cpu_temp)
            _history["gpu_temp"].append(s.gpu_temp)
            _history["soc_temp"].append(s.soc_temp)
            _history["total_power"].append(s.total_power)
    except Exception as exc:
        logger.debug("push_reading error: %s", exc)


def _build_temp_figure() -> Any:
    """Return a Plotly figure of CPU/GPU/SoC temperatures over time."""
    import plotly.graph_objects as go

    _push_reading()

    with _lock:
        ts    = list(_history["ts"])
        cpu   = list(_history["cpu_temp"])
        gpu   = list(_history["gpu_temp"])
        soc   = list(_history["soc_temp"])

    if not ts:
        fig = go.Figure()
        fig.update_layout(
            title="Waiting for thermal data...",
            template="plotly_dark",
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
        )
        return fig

    # Convert epoch to relative seconds
    t0 = ts[0]
    xs = [t - t0 for t in ts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=cpu, name="CPU", line=dict(color="#f59e0b", width=2)))
    fig.add_trace(go.Scatter(x=xs, y=gpu, name="GPU", line=dict(color="#6366f1", width=2)))
    fig.add_trace(go.Scatter(x=xs, y=soc, name="SoC", line=dict(color="#22c55e", width=2)))

    # Warning / critical threshold lines
    fig.add_hline(y=85, line_dash="dash", line_color="#ef4444",
                  annotation_text="Critical 85°C", annotation_position="top right")
    fig.add_hline(y=72, line_dash="dot", line_color="#f59e0b",
                  annotation_text="Warm 72°C", annotation_position="top right")

    fig.update_layout(
        title="Thermal History",
        xaxis_title="Elapsed (s)",
        yaxis_title="°C",
        template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=50, b=40),
        height=280,
    )
    return fig


def _build_power_figure() -> Any:
    """Return a Plotly figure of total power draw over time."""
    import plotly.graph_objects as go

    with _lock:
        ts    = list(_history["ts"])
        power = list(_history["total_power"])

    if not ts:
        return go.Figure()

    t0 = ts[0]
    xs = [t - t0 for t in ts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=power, fill="tozeroy", name="Power (W)",
                             line=dict(color="#6366f1", width=2)))
    fig.update_layout(
        title="Power Draw",
        xaxis_title="Elapsed (s)",
        yaxis_title="Watts",
        template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#e2e8f0"),
        margin=dict(l=50, r=20, t=50, b=40),
        height=200,
    )
    return fig


def render_thermal_chart() -> None:
    """Render live thermal charts inside the current gr.Blocks context."""
    gr.Markdown("### Thermal Time-Series")

    with gr.Row():
        temp_plot = gr.Plot(
            value=_build_temp_figure,
            label="Temperature History",
            every=10,
            show_label=False,
        )
        power_plot = gr.Plot(
            value=_build_power_figure,
            label="Power Draw",
            every=10,
            show_label=False,
        )

    clear_btn = gr.Button("🗑 Clear History", size="sm", variant="secondary", scale=0)
    clear_btn.click(
        fn=lambda: (_lock.__enter__() or None, _history["ts"].clear(),  # noqa
                    _history["cpu_temp"].clear(), _history["gpu_temp"].clear(),
                    _history["soc_temp"].clear(), _history["total_power"].clear(),
                    _lock.__exit__(None, None, None),  # noqa
                    _build_temp_figure(), _build_power_figure())[-2:],
        outputs=[temp_plot, power_plot],
    )
