"""Ray-based distributed job orchestration for parallel distillation runs.

Enables running multiple training jobs in parallel (e.g. ablation sweeps,
A/B experiments, multi-student distillation) using Ray remote tasks.
Falls back to sequential execution when Ray is not installed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _has_ray() -> bool:
    try:
        import ray  # type: ignore[import]
        return True
    except ImportError:
        return False


# ── Job descriptor ─────────────────────────────────────────────────────────────

@dataclass
class DistillJob:
    """A single distillation job to be executed."""
    job_id: str
    backend: str
    teacher: str
    student: str
    dataset: str
    output_dir: str
    epochs: int = 3
    lr: float = 2e-4
    batch_size: int = 4
    lora_rank: int = 16
    extra_args: dict[str, Any] = field(default_factory=dict)

    def to_cmd_args(self) -> list[str]:
        """Return distill agent CLI arguments for this job."""
        return [
            "--backend",    self.backend,
            "--teacher",    self.teacher,
            "--student",    self.student,
            "--dataset",    self.dataset,
            "--output_dir", self.output_dir,
            "--epochs",     str(self.epochs),
            "--lr",         str(self.lr),
            "--batch_size", str(self.batch_size),
            "--lora_r",     str(self.lora_rank),
        ]


@dataclass
class JobResult:
    job_id: str
    success: bool
    output_dir: str
    metrics: dict[str, Any]
    elapsed_sec: float
    error: str = ""


# ── Ray remote task ────────────────────────────────────────────────────────────

def _run_job_subprocess(job: DistillJob) -> JobResult:
    """Execute one DistillJob via subprocess (works with or without Ray)."""
    import subprocess
    import sys

    t0 = time.time()
    cmd = [sys.executable, "-m", "distill.orchestration.agent"] + job.to_cmd_args()

    logger.info("[%s] Starting: %s → %s (%s)", job.job_id, job.teacher, job.student, job.backend)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            return JobResult(
                job_id=job.job_id, success=False,
                output_dir=job.output_dir, metrics={},
                elapsed_sec=elapsed, error=result.stderr[-2000:],
            )
        # Try to read metrics from output dir
        metrics = _read_output_metrics(job.output_dir)
        logger.info("[%s] Complete in %.1fs", job.job_id, elapsed)
        return JobResult(
            job_id=job.job_id, success=True,
            output_dir=job.output_dir, metrics=metrics,
            elapsed_sec=elapsed,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("[%s] Failed: %s", job.job_id, exc)
        return JobResult(
            job_id=job.job_id, success=False,
            output_dir=job.output_dir, metrics={},
            elapsed_sec=elapsed, error=str(exc),
        )


def _read_output_metrics(output_dir: str) -> dict[str, Any]:
    import json
    for name in ("metrics.json", "trainer_state.json", "eval_results.json"):
        p = Path(output_dir) / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return {}


# ── Parallel executor ──────────────────────────────────────────────────────────

def run_jobs_parallel(
    jobs: list[DistillJob],
    max_parallel: int = 2,
    progress_cb: Callable[[str, str], None] | None = None,
) -> list[JobResult]:
    """Run a list of DistillJobs in parallel using Ray when available.

    Args:
        jobs:         Jobs to execute.
        max_parallel: Maximum concurrent jobs (ignored when Ray unavailable — runs all).
        progress_cb:  Optional callback(job_id, status) for UI updates.

    Returns:
        List of JobResult in same order as jobs.
    """
    if not jobs:
        return []

    if _has_ray():
        return _run_with_ray(jobs, max_parallel, progress_cb)
    return _run_sequential(jobs, progress_cb)


def _run_with_ray(
    jobs: list[DistillJob],
    max_parallel: int,
    progress_cb: Callable | None,
) -> list[JobResult]:
    import ray  # type: ignore[import]

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    remote_fn = ray.remote(_run_job_subprocess)
    results_map: dict[str, Any] = {}
    pending = list(jobs)
    active: list[tuple[Any, str]] = []  # (future, job_id)
    completed: list[JobResult] = []
    order = {j.job_id: i for i, j in enumerate(jobs)}

    while pending or active:
        while pending and len(active) < max_parallel:
            job = pending.pop(0)
            fut = remote_fn.remote(job)
            active.append((fut, job.job_id))
            if progress_cb:
                progress_cb(job.job_id, "running")

        if active:
            done, active_futures = ray.wait(
                [f for f, _ in active], num_returns=1, timeout=5.0
            )
            remaining = [(f, jid) for f, jid in active if f not in done]
            for fut in done:
                jid = next(jid for f, jid in active if f == fut)
                result: JobResult = ray.get(fut)
                results_map[jid] = result
                if progress_cb:
                    progress_cb(jid, "completed" if result.success else "failed")
            active = remaining

    return [results_map[j.job_id] for j in jobs if j.job_id in results_map]


def _run_sequential(
    jobs: list[DistillJob],
    progress_cb: Callable | None,
) -> list[JobResult]:
    logger.info("Ray not available — running %d job(s) sequentially", len(jobs))
    results = []
    for job in jobs:
        if progress_cb:
            progress_cb(job.job_id, "running")
        result = _run_job_subprocess(job)
        if progress_cb:
            progress_cb(job.job_id, "completed" if result.success else "failed")
        results.append(result)
    return results


# ── Sweep builder ──────────────────────────────────────────────────────────────

def build_sweep(
    base_job: DistillJob,
    param_grid: dict[str, list[Any]],
) -> list[DistillJob]:
    """Generate a grid of jobs from a base config and parameter grid."""
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    sweep_jobs: list[DistillJob] = []

    for i, combo in enumerate(itertools.product(*values)):
        import dataclasses
        overrides = dict(zip(keys, combo))
        job = dataclasses.replace(
            base_job,
            job_id=f"{base_job.job_id}_sweep_{i}",
            output_dir=f"{base_job.output_dir}/sweep_{i}",
            **overrides,
        )
        sweep_jobs.append(job)

    logger.info("Sweep: %d jobs across %s", len(sweep_jobs), keys)
    return sweep_jobs
