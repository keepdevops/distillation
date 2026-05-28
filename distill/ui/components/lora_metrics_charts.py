"""LoRA-specific Plotly charts for the Training Live tab.

Visualises adapter health over training:
  - Adapter norm history (should be non-trivial and stable)
  - Update ratio (||ΔW||/||W₀|| — should be ~1e-3 to 1e-2)
  - Per-layer rank utilisation heatmap
  - Dead layer count over time
"""
from __future__ import annotations

import logging
from collections import deque
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_MAX_HISTORY = 200
_lock = Lock()

# Rolling LoRA metrics history
_history: dict[str, deque] = {
    "steps":       deque(maxlen=_MAX_HISTORY),
    "norm":        deque(maxlen=_MAX_HISTORY),
    "update_ratio": deque(maxlen=_MAX_HISTORY),
    "grad_norm":   deque(maxlen=_MAX_HISTORY),
    "dead_layers": deque(maxlen=_MAX_HISTORY),
}

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#1e1e2e",
    plot_bgcolor="#1e1e2e",
    font=dict(color="#e2e8f0", size=11),
    margin=dict(l=50, r=20, t=40, b=35),
    height=200,
)


def push_lora_metrics(metrics: dict[str, Any]) -> None:
    """Append one LoRATrainingMetrics dict to the rolling history."""
    with _lock:
        _history["steps"].append(metrics.get("step", 0))
        _history["norm"].append(metrics.get("mean_adapter_norm", 0.0))
        _history["update_ratio"].append(metrics.get("mean_update_ratio", 0.0))
        _history["grad_norm"].append(metrics.get("mean_grad_norm", 0.0))
        _history["dead_layers"].append(float(metrics.get("dead_layers", 0)))


def clear_history() -> None:
    with _lock:
        for d in _history.values():
            d.clear()


def adapter_norm_figure() -> Any:
    """Plotly figure: adapter norm over training steps."""
    import plotly.graph_objects as go

    with _lock:
        steps = list(_history["steps"])
        norms = list(_history["norm"])

    fig = go.Figure()
    if steps:
        fig.add_trace(go.Scatter(
            x=steps, y=norms, name="Adapter Norm",
            line=dict(color="#a78bfa", width=2),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.1)",
        ))
        # Warn threshold
        fig.add_hline(y=1e-6, line_dash="dot", line_color="#ef4444",
                      annotation_text="dead threshold", annotation_position="top right",
                      annotation_font_size=9)
    else:
        fig.add_annotation(text="Waiting for LoRA metrics...",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#475569", size=12))

    fig.update_layout(title="Adapter Norm", xaxis_title="Step",
                      yaxis_title="||W_A||·||W_B||", **_DARK)
    return fig


def update_ratio_figure() -> Any:
    """Plotly figure: update ratio (relative weight change) over steps."""
    import plotly.graph_objects as go

    with _lock:
        steps  = list(_history["steps"])
        ratios = list(_history["update_ratio"])

    fig = go.Figure()
    if steps:
        fig.add_trace(go.Scatter(
            x=steps, y=ratios, name="Update Ratio",
            line=dict(color="#22c55e", width=2),
        ))
        # Healthy range bands
        fig.add_hrect(y0=1e-4, y1=1e-2, fillcolor="rgba(34,197,94,0.07)",
                      line_width=0, annotation_text="healthy range",
                      annotation_font_size=9)
    else:
        fig.add_annotation(text="Waiting for LoRA metrics...",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#475569", size=12))

    fig.update_layout(title="Update Ratio (||ΔW||/||W₀||)",
                      xaxis_title="Step", yaxis_title="Ratio",
                      yaxis_type="log", **_DARK)
    return fig


def dead_layers_figure() -> Any:
    """Plotly figure: dead (saturated/vanished) layer count over steps."""
    import plotly.graph_objects as go

    with _lock:
        steps = list(_history["steps"])
        dead  = list(_history["dead_layers"])

    fig = go.Figure()
    if steps:
        color = "#ef4444" if (dead and max(dead) > 0) else "#22c55e"
        fig.add_trace(go.Bar(
            x=steps, y=dead, name="Dead Layers",
            marker_color=color, opacity=0.7,
        ))
    fig.update_layout(title="Dead LoRA Layers", xaxis_title="Step",
                      yaxis_title="Count", **_DARK)
    return fig


def health_summary_html() -> str:
    """Return an HTML health card for the current LoRA adapter state."""
    with _lock:
        norms  = list(_history["norm"])
        ratios = list(_history["update_ratio"])
        dead   = list(_history["dead_layers"])

    if not norms:
        return '<span class="pill pill-gray">○ No LoRA metrics yet</span>'

    ok_norm  = norms[-1]  > 1e-6
    ok_ratio = ratios[-1] > 1e-5 if ratios else True
    ok_dead  = (dead[-1] == 0) if dead else True

    if ok_norm and ok_ratio and ok_dead:
        return '<span class="pill pill-green">✅ Adapters healthy</span>'
    problems = []
    if not ok_norm:  problems.append("low norm")
    if not ok_ratio: problems.append("tiny updates")
    if not ok_dead:  problems.append(f"{int(dead[-1])} dead layers")
    return (
        f'<span class="pill pill-yellow">⚠ {", ".join(problems)}</span>'
    )


def render_lora_charts() -> None:
    """Render all LoRA charts inside the current gr.Blocks context."""
    import gradio as gr

    gr.Markdown("### LoRA Adapter Health")
    health = gr.HTML(value=health_summary_html, every=5)

    with gr.Row():
        norm_plot  = gr.Plot(value=adapter_norm_figure,  every=5, label="Adapter Norm")
        ratio_plot = gr.Plot(value=update_ratio_figure,  every=5, label="Update Ratio")
        dead_plot  = gr.Plot(value=dead_layers_figure,   every=5, label="Dead Layers")
