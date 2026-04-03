"""Write trainer_state.json in HuggingFace Trainer format.

Keeps training_watchdog.py compatible with MLX runs — watchdog reads
log_history from this file for plateau detection.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def write_trainer_state(state_path: Path, log_history: list) -> None:
    """Atomically write trainer_state.json with the given log_history.

    Uses a .tmp file + rename to avoid partial writes being read by the watchdog.
    """
    state = {"log_history": log_history}
    tmp = state_path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(state, f)
        tmp.replace(state_path)
    except OSError as e:
        LOG.error("Failed to write trainer_state.json to %s: %s", state_path, e)
        raise
