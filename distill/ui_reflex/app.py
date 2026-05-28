"""Reflex app entry point for the Wow Sausage Maker control center.

Usage:
    reflex run                    # dev mode with hot reload
    reflex run --env prod         # production mode
    python -m distill.ui_reflex.app  # check installation

The Gradio UI remains fully functional at localhost:7860.
This Reflex app runs at localhost:3000 as an alternative reactive frontend.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

try:
    import reflex as rx  # type: ignore[import]
    _HAS_REFLEX = True
except ImportError:
    _HAS_REFLEX = False


if _HAS_REFLEX:
    from distill.ui_reflex.pages.dashboard import dashboard_page, on_load
    from distill.ui_reflex.state.app_state import AppState

    app = rx.App(
        theme=rx.theme(
            appearance="dark",
            accent_color="indigo",
            radius="medium",
            scaling="100%",
        ),
    )

    app.add_page(
        dashboard_page,
        route="/",
        title="Wow Sausage Maker",
        description="LLM Distillation Control Center",
        on_load=on_load(),
    )


def check_install() -> None:
    """Print installation status and quickstart instructions."""
    if not _HAS_REFLEX:
        print("⚠  Reflex is not installed.")
        print("   Install it: pip install reflex")
        print("   Gradio UI still works: python -m distill.ui.app")
        return

    print("✅ Reflex installed — Wow Sausage Maker Reflex app ready.")
    print()
    print("Start the reactive UI:")
    print("   cd /path/to/distill")
    print("   reflex run")
    print()
    print("The Reflex app serves at http://localhost:3000")
    print("The Gradio app serves at http://localhost:7860 (python -m distill.ui.app)")


if __name__ == "__main__":
    check_install()
