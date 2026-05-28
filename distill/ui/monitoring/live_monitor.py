"""Live training monitor — background thread that polls metrics and logs.

Provides a thread-safe store that Gradio tabs can poll via gr.State / every=N.
The monitor attaches to a running subprocess job by watching its output file
or the metrics JSON written by the training loop.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from distill.ui.state_manager import get_job, JobState
from distill.ui.components.log_parser import parse_line, extract_series, smooth

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 2.0  # seconds


class LiveMonitor:
    """Watches a log file and job metrics, updating the global JobState."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._log_path: Path | None = None
        self._metrics_path: Path | None = None
        self._callbacks: list[Callable[[JobState], None]] = []
        self._lock = threading.Lock()
        # Parsed metrics series
        self._steps: list[int] = []
        self._loss: list[float] = []
        self._lr: list[float] = []
        self._grad: list[float] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self, log_path: str | None = None, metrics_path: str | None = None) -> None:
        """Start background polling."""
        self._log_path = Path(log_path) if log_path else None
        self._metrics_path = Path(metrics_path) if metrics_path else None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="LiveMonitor")
        self._thread.start()
        logger.info("LiveMonitor started (log=%s, metrics=%s)", log_path, metrics_path)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("LiveMonitor stopped")

    def on_update(self, cb: Callable[[JobState], None]) -> None:
        """Register a callback invoked on each poll with the current JobState."""
        self._callbacks.append(cb)

    def get_series(self) -> dict[str, list]:
        """Return the current metrics series (thread-safe copy)."""
        with self._lock:
            return {
                "steps": list(self._steps),
                "loss":  list(self._loss),
                "lr":    list(self._lr),
                "grad":  list(self._grad),
                "smoothed": smooth(list(self._loss), window=10),
            }

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Background loop ────────────────────────────────────────────────────

    def _loop(self) -> None:
        log_pos = 0
        while not self._stop.is_set():
            try:
                log_pos = self._poll_log(log_pos)
                self._poll_metrics()
                job = get_job()
                for cb in self._callbacks:
                    try:
                        cb(job)
                    except Exception as exc:
                        logger.debug("monitor callback error: %s", exc)
            except Exception as exc:
                logger.warning("LiveMonitor poll error: %s", exc)
            self._stop.wait(timeout=_POLL_INTERVAL)

    def _poll_log(self, pos: int) -> int:
        """Read new lines from the log file since last position."""
        if not self._log_path or not self._log_path.exists():
            return pos
        try:
            with open(self._log_path) as f:
                f.seek(pos)
                new_lines = f.readlines()
                pos = f.tell()

            job = get_job()
            for line in new_lines:
                job.append_log(line.rstrip())
                rec = parse_line(line)
                if rec:
                    job.update_from_metrics(rec)
                    with self._lock:
                        step = rec.get("step", 0)
                        if step > 0:
                            self._steps.append(step)
                            self._loss.append(rec.get("loss", 0.0))
                            self._lr.append(rec.get("learning_rate", 0.0))
                            self._grad.append(rec.get("grad_norm", 0.0))
        except Exception as exc:
            logger.debug("log poll error: %s", exc)
        return pos

    def _poll_metrics(self) -> None:
        """Read trainer_state.json written by HF Trainer."""
        if not self._metrics_path or not self._metrics_path.exists():
            return
        try:
            import json
            data = json.loads(self._metrics_path.read_text())
            history = data.get("log_history", [])
            if not history:
                return
            with self._lock:
                self._steps.clear()
                self._loss.clear()
                self._lr.clear()
                self._grad.clear()
                for entry in history:
                    step = entry.get("step")
                    loss = entry.get("loss", entry.get("train_loss"))
                    if step is not None and loss is not None:
                        self._steps.append(int(step))
                        self._loss.append(float(loss))
                        self._lr.append(float(entry.get("learning_rate", 0.0)))
                        self._grad.append(float(entry.get("grad_norm", 0.0)))
        except Exception as exc:
            logger.debug("metrics poll error: %s", exc)


# ── Module-level singleton ─────────────────────────────────────────────────────

_monitor = LiveMonitor()


def get_monitor() -> LiveMonitor:
    return _monitor
