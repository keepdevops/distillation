"""Help tab widget layout — delegates each accordion to a sub-module."""
from __future__ import annotations

import base64
import logging

import gradio as gr

from ._help_golden import build_section as _golden
from ._help_configure import build_section as _configure
from ._help_data import build_section as _data
from ._help_eval import build_section as _eval
from ._help_expert import build_section as _expert
from ._help_logs import build_section as _logs
from ._help_faq import build_section as _faq

logger = logging.getLogger(__name__)


def build_tab_help():
    """Build the Help tab.

    Returns
    -------
    Empty dict — the Help tab contains no interactive widgets that need wiring.
    """
    with gr.Tab("Help"):
        gr.Markdown("# Distillation Launcher — Operation Guide")
        gr.Markdown(
            "**Seven tabs:** Configure & Launch · Data Prep · Domain Synthesis · "
            "Eval · **Expert Pipeline** · Live Logs · Help.  "
            "A progress bar appears on every tab — no need to switch to Live Logs to monitor a run. "
            "Click a section below to expand it."
        )

        _golden()
        _configure()
        _data()
        _eval()
        _expert()
        _logs()
        _faq()

        gr.Markdown("---\n### Algorithm Reference")
        try:
            from ...ui.show_algorithms import ALGORITHMS, build_html as _build_html_help
            _help_html = _build_html_help(ALGORITHMS)
            _help_b64 = base64.b64encode(_help_html.encode("utf-8")).decode("ascii")
            gr.HTML(
                f'<div style="border-radius:10px;overflow:hidden;border:1px solid #2a2d3e;">'
                f'<iframe src="data:text/html;base64,{_help_b64}" '
                f'style="width:100%;height:80vh;border:none;" '
                f'sandbox="allow-scripts"></iframe>'
                f'</div>'
            )
        except Exception as _e:
            logger.error("Could not load algorithms for Help tab: %s", _e)
            gr.Markdown(f"⚠️ Could not load algorithms: `{_e}`")

    return {}
