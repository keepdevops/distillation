"""Regex parsing, metrics extraction, progress bar HTML for the launch UI runner."""
from __future__ import annotations

import re

import pandas as pd

_LOG_STEP_RE   = re.compile(r"(?<!\w)step=(\d+)")
_LOG_EPOCH_RE  = re.compile(r"(?<!\w)epoch=([\d.]+)")
_LOG_LOSS_RE   = re.compile(r"(?<!\w)loss=([\d.eE+\-]+)")
_LOG_ELOSS_RE  = re.compile(r"(?<!\w)eval_loss=([\d.eE+\-]+)")
_LOG_GRAD_RE   = re.compile(r"['\"]grad_norm['\"]\s*:\s*['\"]?([\d.eE+\-]+)")
_LOG_LR_RE     = re.compile(r"['\"]learning_rate['\"]\s*:\s*['\"]?([\d.eE+\-]+)")
_LOG_PT_LOSS   = re.compile(r"['\"]loss['\"]\s*:\s*['\"]?([\d.eE+\-]+)")
_LOG_PT_EPOCH  = re.compile(r"['\"]epoch['\"]\s*:\s*['\"]?([\d.]+)")


def parse_line_to_json(line: str) -> dict | None:
    """Extract structured metrics from a log line. Returns None if no metrics found."""
    import time as _t
    entry: dict = {}
    m = _LOG_STEP_RE.search(line)
    if m:
        entry["step"] = int(m.group(1))
    m = _LOG_EPOCH_RE.search(line)
    if m:
        entry["epoch"] = float(m.group(1))
    m = _LOG_ELOSS_RE.search(line)
    if m:
        entry["eval_loss"] = float(m.group(1))
    elif (m := _LOG_LOSS_RE.search(line)):
        entry["loss"] = float(m.group(1))
    if "loss" not in entry and "eval_loss" not in entry:
        m = _LOG_PT_LOSS.search(line)
        if m:
            entry["loss"] = float(m.group(1))
        m = _LOG_PT_EPOCH.search(line)
        if m:
            entry["epoch"] = float(m.group(1))
    if not entry:
        return None
    m = _LOG_GRAD_RE.search(line)
    if m:
        entry["grad_norm"] = float(m.group(1))
    m = _LOG_LR_RE.search(line)
    if m:
        entry["lr"] = float(m.group(1))
    entry["ts"] = _t.strftime("%Y-%m-%dT%H:%M:%S")
    entry["msg"] = line.strip()
    return entry


def parse_progress_from_log(log_text: str) -> tuple[float, str]:
    """Return (fraction 0–1, label) from the last progress indicator in log_text."""
    m = list(re.finditer(r'\b(\d{1,3})%\|', log_text))
    if m:
        pct = int(m[-1].group(1))
        return pct / 100, f"{pct}%"
    m = list(re.finditer(r'[Ss]tep\s+(\d+)\s*/\s*(\d+)', log_text))
    if m:
        x, y = int(m[-1].group(1)), int(m[-1].group(2))
        return (x / y if y > 0 else 0.0), f"Step {x}/{y}"
    m = list(re.finditer(r'[Ee]poch[:\s]+(\d+)\s*/\s*(\d+)', log_text))
    if m:
        x, y = int(m[-1].group(1)), int(m[-1].group(2))
        return (x / y if y > 0 else 0.0), f"Epoch {x}/{y}"
    m = list(re.finditer(r'[Pp]rogress[:\s]+(\d+)%', log_text))
    if m:
        pct = int(m[-1].group(1))
        return pct / 100, f"{pct}%"
    return 0.0, ""


def extract_metrics_from_lines(lines: list[str], loss_step_counter: int) -> tuple[list[dict], int]:
    """Parse loss/grad_norm from trainer log lines. Returns (metrics_list, updated_counter)."""
    results = []
    for line in lines:
        if "eval_loss" in line and "loss=" not in line.replace("eval_loss", ""):
            continue

        loss_m = re.search(r"(?<!['\"\w])loss=([\d.eE+\-]+)", line)
        if not loss_m:
            loss_m = re.search(r"['\"]loss['\"]\s*:\s*['\"]?([\d.eE+\-]+)['\"]?", line)
        if not loss_m:
            continue
        try:
            loss = float(loss_m.group(1))
        except ValueError:
            continue

        loss_step_counter += 1
        step_m = re.search(r"(?<!['\"\w])step=(\d+)", line) or \
                 re.search(r"['\"]step['\"]\s*:\s*(\d+)", line)
        epoch_m = re.search(r"(?<!['\"\w])epoch=([\d.]+)", line) or \
                  re.search(r"['\"]epoch['\"]\s*:\s*['\"]?([\d.]+)['\"]?", line)
        grad_m = re.search(r"['\"]grad_norm['\"]\s*:\s*['\"]?([\d.eE+\-]+)['\"]?", line)

        step = int(step_m.group(1)) if step_m else loss_step_counter
        entry: dict = {"step": step, "loss": loss}
        if epoch_m:
            entry["epoch"] = float(epoch_m.group(1))
        if grad_m:
            try:
                entry["grad_norm"] = float(grad_m.group(1))
            except ValueError:
                pass
        results.append(entry)
    return results, loss_step_counter


def loss_plot_data(loss_history: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not loss_history:
        return pd.DataFrame({"step": [0], "loss": [0.0]}), pd.DataFrame({"step": [0], "grad_norm": [0.0]})
    loss_rows = [{"step": e["step"], "loss": e["loss"]} for e in loss_history]
    grad_rows = [{"step": e["step"], "grad_norm": e["grad_norm"]}
                 for e in loss_history if "grad_norm" in e]
    return pd.DataFrame(loss_rows), pd.DataFrame(grad_rows) if grad_rows else pd.DataFrame({"step": [], "grad_norm": []})


def progress_bar_html(fraction: float, label: str, running: bool = True) -> str:
    """Return an HTML snippet for a compact progress bar, or '' when idle."""
    if not running and not label:
        return ""
    pct = max(0, min(100, int(fraction * 100)))
    color = "#2563eb" if running else "#16a34a"
    status = label or ("Running…" if running else "Done")
    return (
        '<div style="margin:4px 0 6px;">'
        '<div style="display:flex;justify-content:space-between;font-size:11px;'
        'color:#6b7280;margin-bottom:2px;">'
        f'<span>{status}</span><span>{pct}%</span></div>'
        '<div style="background:#e5e7eb;border-radius:3px;height:6px;overflow:hidden;">'
        f'<div style="width:{pct}%;background:{color};height:6px;'
        'border-radius:3px;transition:width 0.3s ease;"></div>'
        '</div></div>'
    )
