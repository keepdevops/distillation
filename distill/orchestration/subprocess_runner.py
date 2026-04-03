"""Subprocess execution and log parsing utilities for the distillation agent."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path

LOG = logging.getLogger(__name__)

_STEP_RE    = re.compile(r"(?<!\w)step=(\d+)")
_EPOCH_RE   = re.compile(r"(?<!\w)epoch=([\d.]+)")
_LOSS_RE    = re.compile(r"(?<!\w)loss=([\d.eE+\-]+)")
_ELOSS_RE   = re.compile(r"(?<!\w)eval_loss=([\d.eE+\-]+)")
_GRAD_RE    = re.compile(r"['\"]grad_norm['\"]\s*:\s*['\"]?([\d.eE+\-]+)")
_LR_RE      = re.compile(r"['\"]learning_rate['\"]\s*:\s*['\"]?([\d.eE+\-]+)")
_PT_LOSS_RE = re.compile(r"['\"]loss['\"]\s*:\s*['\"]?([\d.eE+\-]+)")
_PT_EPOCH_RE = re.compile(r"['\"]epoch['\"]\s*:\s*['\"]?([\d.]+)")


def parse_log_line(line: str) -> dict | None:
    """Extract structured metrics from a single log line. Returns None if no metrics found."""
    import time as _time
    entry: dict = {}

    m = _STEP_RE.search(line)
    if m:
        entry["step"] = int(m.group(1))
    m = _EPOCH_RE.search(line)
    if m:
        entry["epoch"] = float(m.group(1))
    m = _ELOSS_RE.search(line)
    if m:
        entry["eval_loss"] = float(m.group(1))
    elif (m := _LOSS_RE.search(line)):
        entry["loss"] = float(m.group(1))

    if "loss" not in entry and "eval_loss" not in entry:
        m = _PT_LOSS_RE.search(line)
        if m:
            entry["loss"] = float(m.group(1))
        m = _PT_EPOCH_RE.search(line)
        if m:
            entry["epoch"] = float(m.group(1))

    if not entry:
        return None

    m = _GRAD_RE.search(line)
    if m:
        entry["grad_norm"] = float(m.group(1))
    m = _LR_RE.search(line)
    if m:
        entry["lr"] = float(m.group(1))
    entry["ts"] = _time.strftime("%Y-%m-%dT%H:%M:%S")
    return entry


def run_cmd(cmd: list[str], cwd: Path, env: dict | None = None,
            json_log: Path | None = None) -> None:
    """Run command with live stdout streaming; raise on non-zero exit.

    If json_log is given, structured metrics parsed from each line are
    appended as JSON Lines to that file alongside the plain text log.
    """
    LOG.info("Running: %s", " ".join(cmd))
    env = env or os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    jf = open(json_log, "a") if json_log else None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                LOG.info("[subprocess] %s", line)
                if jf:
                    entry = parse_log_line(line)
                    if entry:
                        jf.write(json.dumps(entry) + "\n")
                        jf.flush()
    finally:
        if jf:
            jf.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}"
        )


def find_llama_cpp(project_root: Path) -> Path | None:
    """Locate the llama.cpp directory relative to the project root."""
    candidates = [
        Path("/Users/Shared/llama"),
        project_root / "llama.cpp",
        project_root.parent / "llama.cpp",
    ]
    for candidate in candidates:
        if (candidate / "convert_hf_to_gguf.py").exists():
            return candidate
    return None
