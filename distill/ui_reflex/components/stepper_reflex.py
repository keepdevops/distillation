"""Reactive pipeline stepper component for Reflex.

Renders a vertical step list that reacts to AppState.job_phase.
Falls back to a static HTML version when Reflex is not installed.
"""
from __future__ import annotations

from typing import Any

try:
    import reflex as rx  # type: ignore[import]
    _HAS_REFLEX = True
except ImportError:
    _HAS_REFLEX = False

PIPELINE_STEPS = [
    ("filter",   "Data Filter"),
    ("synth",    "Synthetic Data"),
    ("sft",      "SFT Warmup"),
    ("distill",  "Distillation"),
    ("align",    "Alignment"),
    ("eval",     "Evaluation"),
    ("export",   "Export"),
]

_STATUS_COLORS = {
    "done":    "#22c55e",
    "active":  "#6366f1",
    "pending": "#475569",
    "error":   "#ef4444",
}


def _step_status(step_id: str, active_phase: str) -> str:
    """Return 'done' / 'active' / 'pending' / 'error' for a step."""
    phase_order = [s[0] for s in PIPELINE_STEPS]
    try:
        active_idx = phase_order.index(active_phase)
        step_idx   = phase_order.index(step_id)
        if step_idx < active_idx:
            return "done"
        if step_idx == active_idx:
            return "active"
    except ValueError:
        pass
    return "pending"


if _HAS_REFLEX:
    from distill.ui_reflex.state.app_state import AppState

    def stepper_reflex() -> rx.Component:
        """Return a reactive Reflex stepper component."""

        def step_item(step_id: str, label: str) -> rx.Component:
            return rx.box(
                rx.hstack(
                    rx.cond(
                        AppState.job_phase == step_id,
                        rx.spinner(size="2"),
                        rx.text("●", color=rx.cond(
                            AppState.job_phase > step_id, "#22c55e", "#475569"
                        )),
                    ),
                    rx.text(label, font_weight="600", font_size="0.875rem"),
                    spacing="2",
                    align="center",
                ),
                padding="0.4rem 0.75rem",
                border_radius="0.375rem",
                background=rx.cond(
                    AppState.job_phase == step_id,
                    "rgba(99,102,241,0.15)",
                    "transparent",
                ),
                border_left=rx.cond(
                    AppState.job_phase == step_id,
                    "3px solid #6366f1",
                    "3px solid transparent",
                ),
            )

        return rx.vstack(
            *[step_item(sid, label) for sid, label in PIPELINE_STEPS],
            spacing="1",
            width="200px",
        )

    def progress_bar_reflex() -> rx.Component:
        """Reactive progress bar bound to AppState.progress_pct."""
        return rx.box(
            rx.box(
                width=AppState.progress_pct.to_string() + "%",
                height="6px",
                background="#6366f1",
                border_radius="3px",
                transition="width 0.4s ease",
            ),
            width="100%",
            height="6px",
            background="#1e293b",
            border_radius="3px",
            overflow="hidden",
        )

else:
    # Static fallback used in Gradio mode / when Reflex not installed

    def stepper_html(active_phase: str = "") -> str:
        """Return static HTML stepper for current pipeline phase."""
        items = []
        for step_id, label in PIPELINE_STEPS:
            status = _step_status(step_id, active_phase)
            color  = _STATUS_COLORS[status]
            icon   = {"done": "✅", "active": "🔄", "pending": "⬜", "error": "❌"}[status]
            border = f"3px solid {color}" if status == "active" else "3px solid transparent"
            bg     = "rgba(99,102,241,0.12)" if status == "active" else "transparent"
            items.append(
                f'<div style="padding:.35rem .75rem;border-radius:.375rem;'
                f'background:{bg};border-left:{border};display:flex;gap:.5rem;align-items:center;'
                f'margin:.1rem 0">'
                f'  <span>{icon}</span>'
                f'  <span style="color:{color};font-weight:600;font-size:.85rem">{label}</span>'
                f'</div>'
            )
        return '<div style="min-width:180px">' + "".join(items) + "</div>"

    # Stub so callers can always import stepper_reflex
    def stepper_reflex() -> Any:  # type: ignore[misc]
        return None
