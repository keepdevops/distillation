"""Plotly training loss and perplexity chart builders.

All functions return Plotly figures suitable for gr.Plot(). They are kept
separate from the tabs so they can be reused in both the Training Live tab
and the Eval Comparison tab.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1e1e2e",
    plot_bgcolor="#1e1e2e",
    font=dict(color="#e2e8f0", size=11),
    margin=dict(l=50, r=20, t=45, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)


def loss_figure(
    steps: list[int],
    train_loss: list[float],
    eval_loss: list[float] | None = None,
    smoothed: list[float] | None = None,
    title: str = "Training Loss",
) -> Any:
    """Return a Plotly figure of training (+ optional eval) loss."""
    import plotly.graph_objects as go

    fig = go.Figure()

    if steps and train_loss:
        fig.add_trace(go.Scatter(
            x=steps, y=train_loss,
            name="Train Loss",
            line=dict(color="#6366f1", width=1.5),
            opacity=0.6,
        ))

    if smoothed and steps:
        fig.add_trace(go.Scatter(
            x=steps, y=smoothed,
            name="Smoothed",
            line=dict(color="#a5b4fc", width=2.5),
        ))

    if eval_loss and steps:
        fig.add_trace(go.Scatter(
            x=steps[:len(eval_loss)], y=eval_loss,
            name="Eval Loss",
            mode="lines+markers",
            line=dict(color="#22c55e", width=2),
            marker=dict(size=5),
        ))

    if not steps:
        fig.add_annotation(
            text="Waiting for training data...",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#475569", size=14),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Loss",
        height=260,
        **_DARK_LAYOUT,
    )
    return fig


def lr_figure(steps: list[int], lr_vals: list[float]) -> Any:
    """Return a small learning-rate schedule figure."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if steps and lr_vals:
        fig.add_trace(go.Scatter(
            x=steps, y=lr_vals,
            name="LR",
            fill="tozeroy",
            line=dict(color="#f59e0b", width=2),
        ))
    fig.update_layout(
        title="Learning Rate",
        xaxis_title="Step",
        yaxis_title="LR",
        height=160,
        **_DARK_LAYOUT,
    )
    return fig


def grad_norm_figure(steps: list[int], grad_vals: list[float]) -> Any:
    """Return a gradient norm figure."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if steps and grad_vals:
        fig.add_trace(go.Scatter(
            x=steps, y=grad_vals,
            name="Grad Norm",
            line=dict(color="#ef4444", width=1.5),
        ))
    fig.update_layout(
        title="Gradient Norm",
        xaxis_title="Step",
        yaxis_title="|grad|",
        height=160,
        **_DARK_LAYOUT,
    )
    return fig


def radar_figure(
    models: list[str],
    categories: list[str],
    scores: list[list[float]],
) -> Any:
    """Return a radar chart comparing multiple models across quality dimensions."""
    import plotly.graph_objects as go

    colors = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444"]
    fig = go.Figure()
    for i, (model, vals) in enumerate(zip(models, scores)):
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            name=model,
            fill="toself",
            line=dict(color=colors[i % len(colors)]),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#2a2a3e",
            radialaxis=dict(visible=True, range=[0, 100], color="#94a3b8"),
            angularaxis=dict(color="#94a3b8"),
        ),
        showlegend=True,
        title="Model Quality Radar",
        height=320,
        **_DARK_LAYOUT,
    )
    return fig
