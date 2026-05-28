"""Checkpoint resume detection — find interrupted runs and offer to continue.

Scans output directories for partial checkpoints written by HF Trainer,
MLX training, or our custom format, and returns the best resume point.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    path: Path
    step: int
    backend: str          # "hf_trainer", "mlx", "unknown"
    loss: float
    is_resumable: bool
    details: dict[str, Any]

    def summary(self) -> str:
        return (
            f"{self.backend} checkpoint at step {self.step} "
            f"(loss={self.loss:.4f}) — {self.path}"
        )


def _parse_hf_checkpoint(ckpt_dir: Path) -> CheckpointInfo | None:
    """Parse a HF Trainer checkpoint directory."""
    state_file = ckpt_dir / "trainer_state.json"
    if not state_file.exists():
        return None
    try:
        state = json.loads(state_file.read_text())
        step = state.get("global_step", 0)
        history = state.get("log_history", [])
        loss = next(
            (e["loss"] for e in reversed(history) if "loss" in e), 0.0
        )
        return CheckpointInfo(
            path=ckpt_dir, step=step, backend="hf_trainer",
            loss=loss, is_resumable=True,
            details={"global_step": step, "epoch": state.get("epoch", 0)},
        )
    except Exception as exc:
        logger.debug("HF checkpoint parse error at %s: %s", ckpt_dir, exc)
        return None


def _parse_mlx_checkpoint(ckpt_dir: Path) -> CheckpointInfo | None:
    """Parse an MLX checkpoint directory (adapters + metrics.json)."""
    metrics_file = ckpt_dir / "metrics.json"
    adapter_file = ckpt_dir / "adapters.npz"
    if not adapter_file.exists():
        return None
    step, loss = 0, 0.0
    if metrics_file.exists():
        try:
            m = json.loads(metrics_file.read_text())
            step = int(m.get("step", 0))
            loss = float(m.get("loss", 0.0))
        except Exception:
            pass
    return CheckpointInfo(
        path=ckpt_dir, step=step, backend="mlx",
        loss=loss, is_resumable=True,
        details={"has_adapters": True},
    )


def find_checkpoints(output_dir: str, max_depth: int = 3) -> list[CheckpointInfo]:
    """Scan output_dir for all resumable checkpoints."""
    root = Path(output_dir)
    if not root.exists():
        return []

    found: list[CheckpointInfo] = []

    # HF Trainer: checkpoint-N directories
    for ckpt in root.rglob("checkpoint-*"):
        if ckpt.is_dir():
            info = _parse_hf_checkpoint(ckpt)
            if info:
                found.append(info)

    # MLX: directories containing adapters.npz
    for ckpt in root.rglob("adapters.npz"):
        info = _parse_mlx_checkpoint(ckpt.parent)
        if info:
            found.append(info)

    # Deduplicate by path
    seen: set[Path] = set()
    unique: list[CheckpointInfo] = []
    for c in found:
        if c.path not in seen:
            seen.add(c.path)
            unique.append(c)

    unique.sort(key=lambda c: c.step, reverse=True)
    logger.info("Found %d checkpoint(s) in %s", len(unique), root)
    return unique


def best_checkpoint(output_dir: str) -> CheckpointInfo | None:
    """Return the highest-step resumable checkpoint, or None."""
    ckpts = find_checkpoints(output_dir)
    return ckpts[0] if ckpts else None


def resume_args(ckpt: CheckpointInfo) -> dict[str, Any]:
    """Return kwargs to pass to a trainer to resume from this checkpoint."""
    if ckpt.backend == "hf_trainer":
        return {"resume_from_checkpoint": str(ckpt.path)}
    if ckpt.backend == "mlx":
        return {"adapter_path": str(ckpt.path)}
    return {}


def checkpoint_status_markdown(output_dir: str) -> str:
    """Return a markdown summary of available checkpoints for the UI."""
    ckpts = find_checkpoints(output_dir)
    if not ckpts:
        return "*No checkpoints found in this output directory.*"

    lines = ["| Backend | Step | Loss | Path |", "|---|---|---|---|"]
    for c in ckpts[:10]:
        lines.append(
            f"| {c.backend} | {c.step} | {c.loss:.4f} | `{c.path.name}` |"
        )
    return "\n".join(lines)
