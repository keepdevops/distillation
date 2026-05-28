"""Background job queue for Reflex — manages training subprocesses.

Provides a thread-safe queue that accepts training job configs, runs them
as subprocesses, and streams status updates back to the Reflex AppState.
"""
from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class QueuedJob:
    job_id: str
    cmd: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    submitted_at: float = field(default_factory=time.time)


@dataclass
class JobUpdate:
    job_id: str
    event: str          # "started" / "log" / "metrics" / "completed" / "failed"
    data: Any = None


class JobQueueManager:
    """Thread-safe manager for training subprocess jobs.

    Usage:
        mgr = JobQueueManager(on_update=my_callback)
        mgr.start()
        mgr.submit(QueuedJob("job1", ["python", "-m", "distill.orchestration.agent", ...]))
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        on_update: Callable[[JobUpdate], None] | None = None,
    ) -> None:
        self._queue: queue.Queue[QueuedJob] = queue.Queue()
        self._active: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._max_concurrent = max_concurrent
        self._on_update = on_update or (lambda u: None)
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background worker thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="JobQueueWorker"
        )
        self._worker_thread.start()
        logger.info("JobQueueManager started (max_concurrent=%d)", self._max_concurrent)

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    def submit(self, job: QueuedJob) -> None:
        self._queue.put(job)
        logger.info("Job queued: %s", job.job_id)

    def submit_distill(
        self,
        job_id: str,
        backend: str,
        teacher: str,
        student: str,
        dataset: str,
        output_dir: str,
        **kwargs: Any,
    ) -> None:
        """Convenience wrapper — builds CLI and submits."""
        cmd = [
            sys.executable, "-m", "distill.orchestration.agent",
            "--backend",    backend,
            "--teacher",    teacher,
            "--student",    student,
            "--dataset",    dataset,
            "--output_dir", output_dir,
        ]
        for k, v in kwargs.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        self.submit(QueuedJob(job_id=job_id, cmd=cmd,
                              metadata={"backend": backend, "teacher": teacher}))

    def cancel(self, job_id: str) -> bool:
        """Send SIGTERM to a running job. Returns True if found."""
        with self._lock:
            proc = self._active.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
            logger.info("Cancelled job: %s", job_id)
            return True
        return False

    def cancel_all(self) -> None:
        with self._lock:
            for jid, proc in list(self._active.items()):
                if proc.poll() is None:
                    proc.terminate()

    def active_jobs(self) -> list[str]:
        with self._lock:
            return [jid for jid, p in self._active.items() if p.poll() is None]

    def queue_size(self) -> int:
        return self._queue.qsize()

    # ── Worker loop ────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            # Start new jobs up to max_concurrent
            while len(self.active_jobs()) < self._max_concurrent:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._launch(job)

            self._stop_event.wait(timeout=1.0)

    def _launch(self, job: QueuedJob) -> None:
        try:
            proc = subprocess.Popen(
                job.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).resolve().parent.parent.parent.parent),
            )
            with self._lock:
                self._active[job.job_id] = proc

            self._on_update(JobUpdate(job.job_id, "started", job.metadata))

            t = threading.Thread(
                target=self._stream_output,
                args=(job.job_id, proc),
                daemon=True,
                name=f"stream-{job.job_id}",
            )
            t.start()
            logger.info("Launched job %s (pid=%d)", job.job_id, proc.pid)
        except Exception as exc:
            logger.error("Launch failed for %s: %s", job.job_id, exc)
            self._on_update(JobUpdate(job.job_id, "failed", str(exc)))

    def _stream_output(self, job_id: str, proc: subprocess.Popen) -> None:
        from distill.ui.components.log_parser import parse_line
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                self._on_update(JobUpdate(job_id, "log", line))
                rec = parse_line(line)
                if rec:
                    self._on_update(JobUpdate(job_id, "metrics", rec))
            proc.wait()
            event = "completed" if proc.returncode == 0 else "failed"
            self._on_update(JobUpdate(job_id, event, proc.returncode))
        except Exception as exc:
            self._on_update(JobUpdate(job_id, "failed", str(exc)))
        finally:
            with self._lock:
                self._active.pop(job_id, None)


# ── Module-level singleton ─────────────────────────────────────────────────────

_manager: JobQueueManager | None = None


def get_manager() -> JobQueueManager:
    global _manager
    if _manager is None:
        _manager = JobQueueManager()
        _manager.start()
    return _manager
