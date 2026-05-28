"""Eval Comparison tab — interactive model comparison table + radar chart.

Loads metrics from experiment_log.jsonl and renders side-by-side comparisons
with a Plotly radar chart showing quality dimensions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

_QUALITY_DIMS = ["Perplexity↓", "Distinct-2", "Quality Score", "Speed (tps)", "Size (B)"]
_HEADERS = ["Model", "Backend", "Perplexity", "Quality", "Speed (tps)", "Params (B)", "Date"]


def _load_comparison_rows() -> list[list[Any]]:
    """Load experiment runs and return table rows."""
    try:
        from distill.ui.experiment_log import ExperimentLog
        log = ExperimentLog()
        runs = log.load_all()
        rows = []
        for r in runs[-20:]:  # last 20 runs
            cfg  = r.get("config", {})
            metrics = r.get("metrics", {})
            rows.append([
                Path(cfg.get("output_dir", "?")).name,
                cfg.get("backend", "?"),
                f'{metrics.get("perplexity", 0.0):.2f}',
                f'{metrics.get("quality_score", 0.0):.3f}',
                f'{metrics.get("tokens_per_sec", 0.0):.1f}',
                f'{metrics.get("param_count", 0) / 1e9:.1f}' if metrics.get("param_count") else "?",
                r.get("timestamp", "")[:10],
            ])
        return rows
    except Exception as exc:
        logger.warning("load_comparison_rows: %s", exc)
        return []


def _build_radar(selected_rows: list[list]) -> Any:
    from distill.ui.components.loss_charts import radar_figure
    if not selected_rows:
        return radar_figure(["No data"], _QUALITY_DIMS, [[0] * len(_QUALITY_DIMS)])

    models, scores = [], []
    for row in selected_rows[:4]:
        name = str(row[0])
        try:
            ppl   = 100 - min(100, float(row[2]))   # invert: lower ppl = better
            qual  = float(row[3]) * 100
            speed = min(100, float(row[4]))
            size_score = max(0, 100 - float(row[5]) * 10) if row[5] != "?" else 50
            models.append(name)
            scores.append([ppl, qual * 10, qual, speed, size_score])
        except (ValueError, IndexError):
            continue

    if not models:
        return radar_figure(["No data"], _QUALITY_DIMS, [[0] * len(_QUALITY_DIMS)])
    return radar_figure(models, _QUALITY_DIMS, scores)


def build_tab() -> None:
    """Render the Eval Comparison tab inside the current gr.Blocks context."""
    gr.Markdown("## 📊 Eval & Model Comparison")
    gr.Markdown(
        "Compare distilled checkpoints across quality, speed, and size dimensions. "
        "Select rows in the table to populate the radar chart."
    )

    with gr.Row():
        refresh_btn = gr.Button("↻ Refresh", variant="secondary", scale=0)

    # ── Comparison table ──────────────────────────────────────────────────
    rows = _load_comparison_rows()
    table = gr.Dataframe(
        value=rows if rows else None,
        headers=_HEADERS,
        label="Model Comparison",
        interactive=False,
        wrap=True,
        datatype=["str"] * len(_HEADERS),
    )

    # ── Radar chart ───────────────────────────────────────────────────────
    gr.Markdown("### Quality Radar")
    radar = gr.Plot(value=lambda: _build_radar([]), label="Radar")

    # ── Detailed metrics for selected row ─────────────────────────────────
    gr.Markdown("### Selected Model Detail")
    detail_md = gr.Markdown("*Click a row in the table.*")

    # ── Regression status ─────────────────────────────────────────────────
    gr.Markdown("### Regression Check")
    regression_md = gr.Markdown(_regression_summary())

    # ── Events ────────────────────────────────────────────────────────────
    def on_select(evt: gr.SelectData, data):
        if data is None or evt.index is None:
            return _build_radar([]), "*No selection.*"
        try:
            row_idx = evt.index[0]
            row = data[row_idx]
            radar_fig = _build_radar([row])
            detail = "\n".join(
                f"**{h}:** {v}" for h, v in zip(_HEADERS, row)
            )
            return radar_fig, detail
        except Exception as exc:
            return _build_radar([]), f"Error: {exc}"

    refresh_btn.click(
        fn=lambda: (
            _load_comparison_rows() or None,
            _regression_summary(),
        ),
        outputs=[table, regression_md],
    )
    table.select(fn=on_select, inputs=[table], outputs=[radar, detail_md])


def _regression_summary() -> str:
    """Compare the two most recent runs and flag regressions."""
    try:
        from distill.ui.experiment_log import ExperimentLog
        runs = ExperimentLog().load_all()
        if len(runs) < 2:
            return "*Need at least 2 runs for regression comparison.*"
        latest, prior = runs[-1], runs[-2]
        lm = latest.get("metrics", {})
        pm = prior.get("metrics", {})

        lines = ["| Metric | Prior | Latest | Delta | Status |", "|---|---|---|---|---|"]
        for key, label, lower_better in [
            ("perplexity", "Perplexity", True),
            ("quality_score", "Quality", False),
            ("tokens_per_sec", "Speed (tps)", False),
        ]:
            lv = lm.get(key)
            pv = pm.get(key)
            if lv is None or pv is None:
                continue
            delta = lv - pv
            if lower_better:
                ok = delta <= 0
            else:
                ok = delta >= 0
            status = "✅" if ok else "⚠️ Regression"
            sign = "+" if delta > 0 else ""
            lines.append(f"| {label} | {pv:.3f} | {lv:.3f} | {sign}{delta:.3f} | {status} |")

        return "\n".join(lines)
    except Exception as exc:
        return f"*Regression check unavailable: {exc}*"
