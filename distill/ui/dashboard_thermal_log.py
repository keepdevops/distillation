"""
Thermal log parsing and plot generation for the dashboard thermal tab.
"""
import csv
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _parse_csv_rows(p: Path):
    """Read CSV rows from thermal log, return (rows, has_temp)."""
    rows = []
    try:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except OSError as exc:
        logger.error("load_thermal OSError reading %s: %s", p, exc)
        return [], False
    if not rows:
        return [], False
    has_temp = "cpu_temp_c" in list(rows[0].keys())
    return rows, has_temp


def _parse_series(rows, has_temp):
    """Convert CSV rows to time-series lists. Returns (times, cpu_t, gpu_t, soc_t, cpu_w, gpu_w, tot_w)."""
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    times, cpu_t, gpu_t, soc_t, cpu_w, gpu_w, tot_w = [], [], [], [], [], [], []
    for row in rows:
        try:
            t = datetime.strptime(row["time"].strip(), "%Y-%m-%d %H:%M:%S")
        except (ValueError, KeyError):
            continue
        times.append(t)
        if has_temp:
            cpu_t.append(_f(row.get("cpu_temp_c", "")))
            gpu_t.append(_f(row.get("gpu_temp_c", "")))
            soc_t.append(_f(row.get("soc_temp_c", "")))
            cpu_w.append(_f(row.get("cpu_power_w", "")))
            gpu_w.append(_f(row.get("gpu_power_w", "")))
            tot_w.append(_f(row.get("total_power_w", "")))
    return times, cpu_t, gpu_t, soc_t, cpu_w, gpu_w, tot_w


def _render_figure(p: Path, has_temp, times, cpu_t, gpu_t, soc_t, cpu_w, gpu_w, tot_w):
    """Build and return a matplotlib figure from parsed thermal series."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    fig.suptitle(f"Thermal — {p.name}", fontsize=11)
    ax0, ax1 = axes
    if has_temp:
        ax0.plot(times, cpu_t, label="CPU", color="steelblue")
        ax0.plot(times, gpu_t, label="GPU", color="tomato")
        ax0.plot(times, soc_t, label="SOC", color="goldenrod", linestyle="--", alpha=0.6)
        ax0.axhline(90, color="red", linestyle=":", linewidth=1, label="pause threshold (90°C)")
    ax0.set_ylabel("°C")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.3)
    if has_temp:
        ax1.plot(times, cpu_w, label="CPU", color="steelblue")
        ax1.plot(times, gpu_w, label="GPU", color="tomato")
        ax1.plot(times, tot_w, label="Total", color="gray", linestyle="--", alpha=0.7)
    ax1.set_ylabel("W")
    ax1.set_xlabel("Time")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    return fig


def load_thermal(log_path):
    """Parse a thermal CSV log and return a matplotlib figure, or None."""
    p = Path(log_path)
    if not p.exists():
        return None
    rows, has_temp = _parse_csv_rows(p)
    if not rows:
        return None
    times, cpu_t, gpu_t, soc_t, cpu_w, gpu_w, tot_w = _parse_series(rows, has_temp)
    if not times:
        return None
    return _render_figure(p, has_temp, times, cpu_t, gpu_t, soc_t, cpu_w, gpu_w, tot_w)
