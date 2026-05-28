"""Regression tracker — auto-compare new checkpoint vs prior best.

Called after every eval run to flag regressions in perplexity, quality,
or speed. Results are stored in the experiment log and surfaced in the
Eval Comparison tab.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Metrics where lower is better
_LOWER_BETTER = {"perplexity", "loss", "eval_loss", "ttft_ms"}

# Default tolerance before flagging a regression (relative %)
_DEFAULT_TOL = 0.02  # 2%


@dataclass
class RegressionResult:
    metric: str
    prior: float
    current: float
    delta: float
    delta_pct: float
    is_regression: bool
    severity: str  # "ok", "warn", "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric":       self.metric,
            "prior":        self.prior,
            "current":      self.current,
            "delta":        self.delta,
            "delta_pct":    self.delta_pct,
            "is_regression": self.is_regression,
            "severity":     self.severity,
        }

    def icon(self) -> str:
        return {"ok": "✅", "warn": "⚠️", "critical": "🔴"}.get(self.severity, "❓")


def check_regression(
    current: dict[str, float],
    prior: dict[str, float],
    tolerance: float = _DEFAULT_TOL,
) -> list[RegressionResult]:
    """Compare current metrics against prior best.

    Args:
        current: Dict of metric_name → value for the new checkpoint.
        prior: Dict of metric_name → value for the reference checkpoint.
        tolerance: Relative tolerance before flagging (e.g. 0.02 = 2%).

    Returns:
        List of RegressionResult, one per compared metric.
    """
    results: list[RegressionResult] = []
    shared = set(current) & set(prior)

    for key in sorted(shared):
        cv = float(current[key])
        pv = float(prior[key])
        if pv == 0.0:
            continue
        delta = cv - pv
        delta_pct = delta / abs(pv)
        lower_better = key in _LOWER_BETTER

        if lower_better:
            is_reg = delta_pct > tolerance       # increase = bad
        else:
            is_reg = delta_pct < -tolerance      # decrease = bad

        if is_reg:
            severity = "critical" if abs(delta_pct) > 0.10 else "warn"
        else:
            severity = "ok"

        results.append(RegressionResult(
            metric=key, prior=pv, current=cv,
            delta=delta, delta_pct=delta_pct * 100,
            is_regression=is_reg, severity=severity,
        ))

    return results


def format_regression_markdown(results: list[RegressionResult]) -> str:
    """Return a markdown table of regression results."""
    if not results:
        return "*No shared metrics to compare.*"

    lines = [
        "| Metric | Prior | Current | Δ% | Status |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        sign = "+" if r.delta_pct >= 0 else ""
        lines.append(
            f"| `{r.metric}` | {r.prior:.4f} | {r.current:.4f} "
            f"| {sign}{r.delta_pct:.1f}% | {r.icon()} {r.severity} |"
        )
    n_reg = sum(1 for r in results if r.is_regression)
    summary = (
        f"\n**{n_reg} regression(s) detected**" if n_reg
        else "\n✅ No regressions detected"
    )
    return "\n".join(lines) + summary


def load_best_metrics(experiment_log_path: str | None = None) -> dict[str, float]:
    """Load the best-known metrics from experiment history."""
    try:
        from distill.ui.experiment_log import ExperimentLog
        from pathlib import Path
        log = ExperimentLog(experiment_log_path or "experiment_log.jsonl")
        runs = log.load_all()
        if not runs:
            return {}
        # Find run with lowest perplexity as the "best"
        best = min(
            (r for r in runs if r.get("metrics")),
            key=lambda r: r["metrics"].get("perplexity", float("inf")),
            default=None,
        )
        return best["metrics"] if best else {}
    except Exception as exc:
        logger.warning("load_best_metrics failed: %s", exc)
        return {}
