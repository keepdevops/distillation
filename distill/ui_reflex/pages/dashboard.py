"""Main Reflex dashboard page — canvas layout with sidebar + content area.

Mirrors the Gradio tab structure but as a single-page reactive app.
Falls back to rendering a status message when Reflex is not installed.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import reflex as rx  # type: ignore[import]
    _HAS_REFLEX = True
except ImportError:
    _HAS_REFLEX = False


if _HAS_REFLEX:
    from distill.ui_reflex.state.app_state import AppState
    from distill.ui_reflex.components.stepper_reflex import stepper_reflex, progress_bar_reflex
    from distill.ui_reflex.components.live_charts import loss_chart_reflex, thermal_row_reflex

    # ── Sidebar ────────────────────────────────────────────────────────────

    def sidebar() -> rx.Component:
        return rx.box(
            rx.vstack(
                rx.text("🌭 Wow Sausage Maker",
                        font_weight="800", font_size="1rem", color="#a5b4fc"),
                rx.divider(border_color="#1e293b"),
                rx.text("Pipeline", font_size="0.7rem", color="#475569",
                        text_transform="uppercase", letter_spacing="0.1em"),
                stepper_reflex(),
                rx.spacer(),
                rx.text("Hardware", font_size="0.7rem", color="#475569",
                        text_transform="uppercase", letter_spacing="0.1em"),
                thermal_row_reflex(),
                spacing="3",
                height="100%",
                align="start",
            ),
            width="240px",
            min_height="100vh",
            background="#0f0f1a",
            border_right="1px solid #1e293b",
            padding="1rem",
            position="fixed",
            left="0",
            top="0",
        )

    # ── Header ─────────────────────────────────────────────────────────────

    def top_bar() -> rx.Component:
        return rx.box(
            rx.hstack(
                rx.text(AppState.job_status, font_size="0.8rem", color="#94a3b8"),
                rx.spacer(),
                rx.badge(AppState.backend, color_scheme="indigo"),
                rx.badge(
                    rx.cond(AppState.is_airgap, "✈ Offline", "🌐 Online"),
                    color_scheme=rx.cond(AppState.is_airgap, "orange", "green"),
                ),
                spacing="3",
                align="center",
                width="100%",
            ),
            padding="0.6rem 1.5rem",
            background="#0f0f1a",
            border_bottom="1px solid #1e293b",
            position="sticky",
            top="0",
            z_index="10",
        )

    # ── Main content ────────────────────────────────────────────────────────

    def training_panel() -> rx.Component:
        return rx.vstack(
            rx.heading("Training Live", size="5", color="#e2e8f0"),
            progress_bar_reflex(),
            rx.hstack(
                rx.text("Step:", font_size="0.8rem", color="#94a3b8"),
                rx.text(AppState.job_step.to_string(), font_weight="600"),
                rx.text("/"),
                rx.text(AppState.job_total.to_string()),
                rx.spacer(),
                rx.text("Loss:", font_size="0.8rem", color="#94a3b8"),
                rx.text(AppState.job_loss.to_string(), font_weight="600", color="#6366f1"),
                spacing="2",
            ),
            loss_chart_reflex(),
            spacing="4",
            width="100%",
            align="start",
        )

    def config_panel() -> rx.Component:
        return rx.vstack(
            rx.heading("Configuration", size="5", color="#e2e8f0"),
            rx.hstack(
                rx.vstack(
                    rx.text("Backend", font_size="0.75rem", color="#94a3b8"),
                    rx.select(
                        ["mlx", "sft", "minillm", "unsloth"],
                        value=AppState.backend,
                        on_change=AppState.set_backend,
                    ),
                    spacing="1",
                ),
                rx.vstack(
                    rx.text("Teacher", font_size="0.75rem", color="#94a3b8"),
                    rx.input(
                        value=AppState.teacher,
                        on_change=AppState.set_teacher,
                        placeholder="HF model ID",
                    ),
                    spacing="1",
                ),
                spacing="4",
                width="100%",
            ),
            rx.code_block(AppState.cli_command, language="bash",
                          font_size="0.75rem", width="100%"),
            spacing="4",
            width="100%",
            align="start",
        )

    # ── Full dashboard page ─────────────────────────────────────────────────

    def dashboard_page() -> rx.Component:
        return rx.box(
            sidebar(),
            rx.box(
                top_bar(),
                rx.box(
                    rx.vstack(
                        config_panel(),
                        rx.divider(border_color="#1e293b"),
                        training_panel(),
                        spacing="6",
                        padding="1.5rem",
                        width="100%",
                    ),
                    margin_left="240px",
                ),
                width="100%",
            ),
            background="#0f172a",
            min_height="100vh",
            font_family="Inter, system-ui, sans-serif",
        )

    def on_load() -> list:
        """Page on_load event handlers."""
        return [AppState.refresh_hardware, AppState.check_airgap]

else:
    # Stubs for import without Reflex

    def dashboard_page() -> str:  # type: ignore[misc]
        return "Reflex not installed. Run: pip install reflex"

    def on_load() -> list:  # type: ignore[misc]
        return []
