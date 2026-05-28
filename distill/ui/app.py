"""Wow Sausage Maker — unified Gradio control center.

Assembles all 13 tabs into a single gr.Blocks() application.
Heavy logic lives in tabs/ and components/; this file only wires them together.
"""
from __future__ import annotations

import logging

import gradio as gr

from distill.ui.theme import build_theme, CUSTOM_CSS
from distill.ui.tab_registry import build_tab
from distill.ui.core.registry import registry  # noqa: F401 — triggers default registration

logger = logging.getLogger(__name__)

# ── Tab manifest ──────────────────────────────────────────────────────────────
# Each entry: (tab_label, module_key)
# module_key is resolved by tab_registry.build_tab()
TAB_MANIFEST = [
    ("⚙ Hardware",          "hardware"),
    ("📦 Data",              "data_prep"),
    ("🎓 SFT",               "sft"),
    ("🧠 Distillation",      "distillation"),
    ("⚖ Alignment",         "alignment"),
    ("📈 Training Live",     "training_live"),
    ("📊 Eval & Compare",    "eval_comparison"),
    ("📋 Eval Config",       "eval"),
    ("💬 Generate",          "generate"),
    ("📤 Export",            "quantize_export"),
    ("🚀 Swarm Export",      "swarm_export"),
    ("▶ Full Auto",         "full_auto_gantt"),
    ("🔬 Experiments",       "experiments"),
    ("📋 Logs",              "logs"),
    ("❓ Help",              "help"),
]


def build_app() -> gr.Blocks:
    """Construct and return the full Gradio app (not yet launched)."""
    theme = build_theme()

    with gr.Blocks(title="Wow Sausage Maker — LLM Distillation") as demo:
        # ── Header ────────────────────────────────────────────────────────
        _render_header()
        _render_global_overlays()

        # ── Tabs ──────────────────────────────────────────────────────────
        with gr.Tabs():
            for label, key in TAB_MANIFEST:
                with gr.TabItem(label):
                    try:
                        build_tab(key)
                    except Exception as exc:
                        logger.error("Failed to build tab '%s': %s", key, exc)
                        gr.Markdown(
                            f"> **Tab unavailable:** `{key}` — {exc}\n\n"
                            "Check logs for details."
                        )

        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML(
            "<div style='text-align:center;color:#475569;font-size:0.75rem;"
            "padding:1rem 0;border-top:1px solid #1e293b;margin-top:1rem'>"
            "Wow Sausage Maker · LLM Distillation Control Center"
            "</div>"
        )

    demo.queue(max_size=20)
    # Gradio 6: theme/css passed to launch(), not Blocks
    demo._theme = theme
    demo._custom_css = CUSTOM_CSS
    return demo


def _render_global_overlays() -> None:
    """Inject keyboard shortcuts and airgap banner (runs once at startup)."""
    try:
        from distill.ui.shortcuts.keyboard_handler import render_keyboard_shortcuts
        render_keyboard_shortcuts()
    except Exception as exc:
        logger.debug("keyboard shortcuts unavailable: %s", exc)
    try:
        from distill.ui.components.airgap_mode import render_airgap_banner, apply_airgap_env
        apply_airgap_env()
        render_airgap_banner()
    except Exception as exc:
        logger.debug("airgap banner unavailable: %s", exc)


def _render_header() -> None:
    """Render the global page header with status pills."""
    try:
        from distill.ui.components.header import render_header
        render_header()
    except Exception as exc:
        logger.warning("Header component unavailable: %s", exc)
        gr.Markdown("## 🌭 Wow Sausage Maker")


# ── Direct launch ─────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point used by distill-ui script and python -m distill.ui.app."""
    import argparse
    from distill.infra.config import cfg

    p = argparse.ArgumentParser(description="Wow Sausage Maker UI")
    p.add_argument("--port", type=int, default=cfg.services.gradio_port)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=app._theme,
        css=app._custom_css,
    )


if __name__ == "__main__":
    main()
