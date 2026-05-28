"""Shared gr.State helpers and session config persistence.

Provides a thin wrapper around Gradio state so tabs can share live training
metrics, session config, and job IDs without coupling to each other directly.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SESSION_DIR = Path.home() / ".distill" / "sessions"


# ── Session config ─────────────────────────────────────────────────────────────

@dataclass
class SessionConfig:
    """Persisted per-session configuration."""
    backend: str = "mlx"
    teacher: str = "Qwen/Qwen2-1.5B-Instruct"
    student: str = "Qwen/Qwen2-0.5B-Instruct"
    dataset: str = "yahma/alpaca-cleaned"
    output_dir: str = "outputs/distilled"
    epochs: int = 3
    lr: float = 2e-4
    batch_size: int = 4
    lora_rank: int = 16
    preset_name: str = ""
    session_id: str = field(default_factory=lambda: f"session_{int(time.time())}")

    def save(self) -> Path:
        _SESSION_DIR.mkdir(parents=True, exist_ok=True)
        path = _SESSION_DIR / f"{self.session_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, session_id: str) -> "SessionConfig":
        path = _SESSION_DIR / f"{session_id}.json"
        if not path.exists():
            return cls(session_id=session_id)
        try:
            data = json.loads(path.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception as exc:
            logger.error("SessionConfig load failed: %s", exc)
            return cls(session_id=session_id)

    @classmethod
    def list_sessions(cls) -> list[str]:
        if not _SESSION_DIR.exists():
            return []
        return sorted(p.stem for p in _SESSION_DIR.glob("session_*.json"))


# ── Live job state ─────────────────────────────────────────────────────────────

@dataclass
class JobState:
    """Tracks a running training job."""
    job_id: str = ""
    status: str = "idle"          # idle / running / paused / completed / failed
    phase: str = ""               # warmup / sft / minillm / eval / export
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    best_loss: float = float("inf")
    log_tail: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    pid: int = 0

    def update_from_metrics(self, metrics: dict[str, Any]) -> None:
        self.step  = int(metrics.get("step", self.step))
        self.loss  = float(metrics.get("loss", self.loss))
        self.phase = str(metrics.get("phase", self.phase))
        if self.loss < self.best_loss:
            self.best_loss = self.loss

    def append_log(self, line: str, max_lines: int = 200) -> None:
        self.log_tail.append(line)
        if len(self.log_tail) > max_lines:
            self.log_tail = self.log_tail[-max_lines:]

    def progress_pct(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(100.0, 100.0 * self.step / self.total_steps)

    def elapsed_str(self) -> str:
        elapsed = time.time() - self.started_at
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── Global singleton (accessed by all tabs in the same process) ────────────────

_global_job = JobState()
_global_config = SessionConfig()


def get_job() -> JobState:
    return _global_job


def get_config() -> SessionConfig:
    return _global_config


def reset_job(job_id: str = "") -> JobState:
    global _global_job
    _global_job = JobState(job_id=job_id or f"job_{int(time.time())}")
    return _global_job


def update_config(**kwargs: Any) -> SessionConfig:
    for k, v in kwargs.items():
        if hasattr(_global_config, k):
            setattr(_global_config, k, v)
    return _global_config
