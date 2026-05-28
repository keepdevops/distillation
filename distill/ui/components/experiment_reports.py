"""Experiment report generator — HTML export of run history.

Produces a self-contained HTML file with training curves, metric tables,
and config summaries for a given run or set of runs.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Distillation Run Report — {title}</title>
<style>
  body {{ font-family: 'Inter', sans-serif; background: #0f172a; color: #e2e8f0;
         max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ color: #a5b4fc; border-bottom: 1px solid #1e293b; padding-bottom: .5rem; }}
  h2 {{ color: #c7d2fe; margin-top: 2rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th {{ background: #1e293b; color: #94a3b8; font-size: .8rem;
        text-transform: uppercase; padding: .5rem; text-align: left; }}
  td {{ padding: .4rem .5rem; border-bottom: 1px solid #1e293b; font-size: .875rem; }}
  tr:hover td {{ background: #1e293b; }}
  .badge {{ display: inline-block; padding: .1rem .5rem; border-radius: 999px;
            font-size: .75rem; font-weight: 600; }}
  .ok   {{ background: rgba(34,197,94,.15); color: #22c55e; }}
  .fail {{ background: rgba(239,68,68,.15);  color: #ef4444; }}
  pre  {{ background: #1e1e2e; padding: 1rem; border-radius: .5rem;
          font-size: .8rem; overflow-x: auto; color: #a5f3fc; }}
  .metric {{ display: inline-block; text-align: center; padding: .75rem 1rem;
             background: #1e293b; border-radius: .5rem; margin: .3rem; }}
  .mv  {{ font-size: 1.6rem; font-weight: 800; color: #6366f1; }}
  .ml  {{ font-size: .7rem; color: #94a3b8; text-transform: uppercase; }}
</style>
</head>
<body>
<h1>🌭 Distillation Run Report</h1>
<p style="color:#94a3b8">{subtitle}</p>
{content}
</body>
</html>"""


def _metric_badge(k: str, v: Any) -> str:
    return f'<div class="metric"><div class="mv">{v}</div><div class="ml">{k}</div></div>'


def _table_from_dict(d: dict, title: str = "") -> str:
    rows = "".join(
        f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
        for k, v in d.items()
    )
    header = f"<h2>{title}</h2>" if title else ""
    return f"{header}<table><tr><th>Key</th><th>Value</th></tr>{rows}</table>"


def build_run_report(run: dict[str, Any]) -> str:
    """Return full HTML string for a single run dict."""
    cfg = run.get("config", {})
    metrics = run.get("metrics", {})
    ts = run.get("timestamp", "")
    run_id = run.get("run_id", "run")

    # Key metric badges
    badge_keys = ["perplexity", "quality_score", "tokens_per_sec", "param_count"]
    badges = ""
    for k in badge_keys:
        v = metrics.get(k)
        if v is not None:
            label = k.replace("_", " ").title()
            fmt = f"{v:.2f}" if isinstance(v, float) else str(v)
            badges += _metric_badge(label, fmt)

    content = (
        f"<div>{badges}</div>"
        + _table_from_dict(cfg, "Configuration")
        + _table_from_dict(metrics, "Metrics")
    )

    log_tail = run.get("log_tail", [])
    if log_tail:
        log_html = "<pre>" + "\n".join(str(l) for l in log_tail[-30:]) + "</pre>"
        content += f"<h2>Log Tail</h2>{log_html}"

    return _HTML_TEMPLATE.format(
        title=run_id,
        subtitle=f"Run: {run_id} · {ts}",
        content=content,
    )


def export_run_to_file(run: dict[str, Any]) -> str:
    """Write a run report to a temp HTML file and return the path."""
    html = build_run_report(run)
    run_id = run.get("run_id", "run").replace("/", "_")
    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{run_id}.html", prefix="distill_report_",
        delete=False, mode="w", encoding="utf-8",
    )
    tmp.write(html)
    tmp.close()
    logger.info("Report written to %s", tmp.name)
    return tmp.name


def export_all_runs(experiment_log_path: str | None = None) -> str:
    """Export all runs in the log to a single HTML file."""
    try:
        from distill.ui.experiment_log import ExperimentLog
        log = ExperimentLog(experiment_log_path or "experiment_log.jsonl")
        runs = log.load_all()
    except Exception as exc:
        logger.error("export_all_runs: %s", exc)
        return ""

    if not runs:
        return ""

    sections = "".join(build_run_report(r) for r in runs[-50:])
    html = _HTML_TEMPLATE.format(
        title="All Runs",
        subtitle=f"{len(runs)} runs exported",
        content=sections,
    )
    tmp = tempfile.NamedTemporaryFile(
        suffix="_all_runs.html", prefix="distill_report_",
        delete=False, mode="w", encoding="utf-8",
    )
    tmp.write(html)
    tmp.close()
    return tmp.name
