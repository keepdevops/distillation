"""Real-time chart components for the Reflex UI.

When Reflex is installed: returns rx.Component trees backed by AppState.
When not installed: returns Plotly figures (used by Gradio mode).

Both paths share the same data model from AppState / LiveMonitor.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import reflex as rx  # type: ignore[import]
    _HAS_REFLEX = True
except ImportError:
    _HAS_REFLEX = False


if _HAS_REFLEX:
    from distill.ui_reflex.state.app_state import AppState

    def loss_chart_reflex() -> rx.Component:
        """Reactive loss chart using Recharts (bundled with Reflex)."""
        return rx.recharts.line_chart(
            rx.recharts.line(
                data_key="loss",
                stroke="#6366f1",
                stroke_width=2,
                dot=False,
            ),
            rx.recharts.x_axis(data_key="step", label="Step"),
            rx.recharts.y_axis(label="Loss"),
            rx.recharts.cartesian_grid(stroke_dasharray="3 3", stroke="#1e293b"),
            rx.recharts.tooltip(
                content_style={"background": "#1e1e2e", "border": "1px solid #1e293b"},
            ),
            data=rx.Var.create(
                [{"step": s, "loss": l}
                 for s, l in zip(AppState.loss_steps, AppState.loss_values)]
            ),
            width="100%",
            height=260,
        )

    def hardware_gauge_reflex(
        label: str,
        value: rx.Var,
        unit: str = "",
        color: str = "#6366f1",
    ) -> rx.Component:
        """A single reactive metric card."""
        return rx.box(
            rx.vstack(
                rx.text(rx.Var.create(value), font_size="1.6rem",
                        font_weight="800", color=color),
                rx.text(unit, font_size="0.7rem", color="#94a3b8"),
                rx.text(label, font_size="0.7rem", color="#94a3b8",
                        text_transform="uppercase", letter_spacing="0.08em"),
                spacing="0",
                align="center",
            ),
            background="#1e293b",
            border_radius="0.5rem",
            padding="0.75rem 1rem",
            min_width="140px",
            text_align="center",
        )

    def thermal_row_reflex() -> rx.Component:
        """Row of reactive hardware metric cards."""
        return rx.hstack(
            hardware_gauge_reflex("CPU", AppState.cpu_temp, "°C", "#f59e0b"),
            hardware_gauge_reflex("GPU", AppState.gpu_temp, "°C", "#6366f1"),
            hardware_gauge_reflex("Power", AppState.total_power, "W", "#22c55e"),
            hardware_gauge_reflex("RAM", AppState.ram_used_gb, "GB", "#06b6d4"),
            spacing="3",
            wrap="wrap",
        )

else:
    # Plotly fallback (used in Gradio mode and CI)

    def loss_chart_reflex() -> Any:  # type: ignore[misc]
        """Return a Plotly figure of the current loss series (Gradio mode)."""
        try:
            from distill.ui.monitoring.live_monitor import get_monitor
            from distill.ui.components.loss_charts import loss_figure
            from distill.ui.components.log_parser import smooth
            series = get_monitor().get_series()
            return loss_figure(
                series["steps"],
                series["loss"],
                smoothed=smooth(series["loss"]),
                title="Training Loss",
            )
        except Exception:
            from distill.ui.components.loss_charts import loss_figure
            return loss_figure([], [], title="Training Loss — waiting...")

    def hardware_gauge_reflex(*_: Any, **__: Any) -> None:  # type: ignore[misc]
        return None

    def thermal_row_reflex() -> str:  # type: ignore[misc]
        """Return HTML gauges (Gradio mode)."""
        try:
            from distill.ui.components.hardware_gauges import build_gauges_html
            return build_gauges_html()
        except Exception:
            return ""
