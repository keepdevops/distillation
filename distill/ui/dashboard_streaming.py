"""Subprocess streaming and progress-bar helpers for the distillation dashboard."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)

def _run_streaming(cmd: list[str]):
    """
    Run a subprocess and yield the accumulated stdout+stderr output after each
    line, so Gradio can stream it live into a Textbox.
    Passes -u to Python for unbuffered output.
    """
    # Insert -u after the Python interpreter for unbuffered output
    if cmd and cmd[0] == sys.executable:
        cmd = [cmd[0], "-u"] + cmd[1:]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=_PROJECT_ROOT,
        )
        output = ""
        for line in proc.stdout:
            output += line
            yield output
        proc.wait()
        suffix = "\n✅ Finished (exit 0)" if proc.returncode == 0 else f"\n⚠ Exit code {proc.returncode}"
        yield output + suffix
    except Exception as e:
        yield f"Failed to start script: {e}\n"


def _parse_progress_from_log(log_text: str) -> tuple[float, str]:
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


def _progress_bar_html(fraction: float, label: str, running: bool = True) -> str:
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


def _is_streaming_done(text: str) -> bool:
    """Return True when _run_streaming has emitted its final terminal line."""
    last = text.rstrip().split('\n')[-1] if text.strip() else ""
    return last.startswith("✅") or last.startswith("⚠ Exit code")


