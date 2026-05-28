"""Log parser — extract structured metrics from training log lines.

Supports:
  - HuggingFace Trainer JSON log lines
  - tqdm-style progress lines ("Step 100/500, loss=1.234")
  - MLX training output ("step 42 | loss 1.2345 | ...")
  - Generic key=value pairs
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Regex patterns ─────────────────────────────────────────────────────────────

_HF_JSON_RE   = re.compile(r'^\{.*"loss".*\}$')
_STEP_LOSS_RE = re.compile(r'[Ss]tep\s+(\d+)(?:/(\d+))?.*?loss[=:\s]+([0-9]+\.[0-9]+)')
_MLX_LINE_RE  = re.compile(
    r'step\s+(\d+)\s*\|\s*loss\s+([0-9]+\.[0-9]+)'
    r'(?:.*?lr\s+([0-9e.+-]+))?'
    r'(?:.*?grad\s+([0-9]+\.[0-9]+))?',
    re.IGNORECASE,
)
_KV_RE        = re.compile(r'(\w+)[=:]\s*([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)')


def parse_line(line: str) -> dict[str, Any] | None:
    """Parse a single log line into a metrics dict, or return None."""
    line = line.strip()
    if not line:
        return None

    # HuggingFace Trainer JSON
    if _HF_JSON_RE.match(line):
        try:
            d = json.loads(line)
            return {
                "step":          int(d.get("step", 0)),
                "loss":          float(d.get("loss", d.get("train_loss", 0.0))),
                "learning_rate": float(d.get("learning_rate", 0.0)),
                "grad_norm":     float(d.get("grad_norm", 0.0)),
                "epoch":         float(d.get("epoch", 0.0)),
                "source":        "hf_trainer",
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # MLX-style: "step 42 | loss 1.2345 | lr 2e-04 | grad 0.91"
    m = _MLX_LINE_RE.search(line)
    if m:
        return {
            "step":          int(m.group(1)),
            "loss":          float(m.group(2)),
            "learning_rate": float(m.group(3)) if m.group(3) else 0.0,
            "grad_norm":     float(m.group(4)) if m.group(4) else 0.0,
            "source":        "mlx",
        }

    # Generic "Step 100/500, loss=1.234"
    m = _STEP_LOSS_RE.search(line)
    if m:
        return {
            "step":       int(m.group(1)),
            "total_steps": int(m.group(2)) if m.group(2) else 0,
            "loss":       float(m.group(3)),
            "source":     "generic",
        }

    return None


def parse_log_file(path: str) -> list[dict[str, Any]]:
    """Parse all metric lines from a log file, returning a sorted list."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    results: list[dict] = []
    try:
        for line in p.read_text(errors="replace").splitlines():
            rec = parse_line(line)
            if rec:
                results.append(rec)
    except Exception as exc:
        logger.error("parse_log_file(%s) failed: %s", path, exc)
    results.sort(key=lambda r: r.get("step", 0))
    return results


def extract_series(records: list[dict]) -> dict[str, list]:
    """Convert a list of metric dicts into per-metric series for Plotly."""
    steps, loss, lr, grad = [], [], [], []
    for r in records:
        if "step" in r and "loss" in r:
            steps.append(r["step"])
            loss.append(r["loss"])
            lr.append(r.get("learning_rate", 0.0))
            grad.append(r.get("grad_norm", 0.0))
    return {"steps": steps, "loss": loss, "lr": lr, "grad_norm": grad}


def smooth(values: list[float], window: int = 10) -> list[float]:
    """Simple moving-average smoothing."""
    if not values or window <= 1:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i + 1]) / (i - start + 1))
    return out
