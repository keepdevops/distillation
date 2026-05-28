"""Experiments tab — run history browser with filters and export.

Wraps distill.ui.experiment_log.ExperimentLog into a Gradio table view.
"""
from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

logger = logging.getLogger(__name__)


def _load_runs() -> tuple[list[list], list[str]]:
    """Return (rows, headers) for the experiments table."""
    try:
        from distill.ui.experiment_log import ExperimentLog
        log = ExperimentLog()
        runs = log.load_all() if hasattr(log, "load_all") else []
        if not runs:
            return [], ["Run ID", "Backend", "Status", "Loss", "Quality", "Date"]
        headers = list(runs[0].keys()) if runs else []
        rows = [[r.get(h, "") for h in headers] for r in runs]
        return rows, headers
    except Exception as exc:
        logger.warning("ExperimentLog load failed: %s", exc)
        return [], ["Run ID", "Backend", "Status", "Loss", "Quality", "Date"]


def build_tab() -> None:
    """Render the Experiments tab inside the current gr.Blocks context."""
    gr.Markdown("## 🔬 Experiments")
    gr.Markdown(
        "Browse, filter, and compare all past distillation runs. "
        "Click a row to see full config and metrics."
    )

    # ── Filter controls ───────────────────────────────────────────────────
    with gr.Row():
        filter_backend = gr.Dropdown(
            choices=["All", "mlx", "sft", "minillm", "unsloth"],
            value="All",
            label="Backend",
            scale=1,
        )
        filter_status = gr.Dropdown(
            choices=["All", "completed", "failed", "running"],
            value="All",
            label="Status",
            scale=1,
        )
        refresh_btn = gr.Button("↻ Refresh", variant="secondary", scale=0)

    # ── Run table ─────────────────────────────────────────────────────────
    rows, headers = _load_runs()
    run_table = gr.Dataframe(
        value=rows if rows else None,
        headers=headers,
        label="Run History",
        interactive=False,
        wrap=True,
        datatype=["str"] * len(headers),
    )

    # ── Selected run detail ───────────────────────────────────────────────
    gr.Markdown("### Selected Run Details")
    run_detail = gr.JSON(label="Full Config & Metrics", value={})
    export_btn = gr.Button("📄 Export as HTML Report", variant="secondary", scale=0)
    export_file = gr.File(label="Download Report", visible=False)

    # ── Event wiring ──────────────────────────────────────────────────────
    def refresh_table(backend, status):
        all_rows, hdrs = _load_runs()
        filtered = all_rows
        if backend != "All":
            idx = hdrs.index("Backend") if "Backend" in hdrs else -1
            if idx >= 0:
                filtered = [r for r in filtered if str(r[idx]).lower() == backend.lower()]
        if status != "All":
            idx = hdrs.index("Status") if "Status" in hdrs else -1
            if idx >= 0:
                filtered = [r for r in filtered if str(r[idx]).lower() == status.lower()]
        return gr.update(value=filtered if filtered else None)

    def show_detail(evt: gr.SelectData, rows):
        if rows is None or evt.index is None:
            return {}
        try:
            row_idx = evt.index[0]
            _, headers = _load_runs()
            row = rows[row_idx]
            return dict(zip(headers, row))
        except Exception as exc:
            return {"error": str(exc)}

    refresh_btn.click(fn=refresh_table, inputs=[filter_backend, filter_status],
                      outputs=run_table)
    filter_backend.change(fn=refresh_table, inputs=[filter_backend, filter_status],
                          outputs=run_table)
    filter_status.change(fn=refresh_table, inputs=[filter_backend, filter_status],
                         outputs=run_table)
    run_table.select(fn=show_detail, inputs=[run_table], outputs=run_detail)
