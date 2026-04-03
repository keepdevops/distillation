"""
Shared loading of HuggingFace Trainer metrics: trainer_state.json + metrics.jsonl.
Also: robust load_trainer_state() for watchdog and dashboards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_trainer_state(output_dir: str | Path):
    """
    Read trainer_state.json. Returns None if missing, invalid JSON, or log_history
    is not a list (same rules as training_watchdog).
    """
    path = Path(output_dir) / "trainer_state.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            state = json.load(f)
        if not isinstance(state.get("log_history"), list):
            logger.warning("trainer_state.json missing or invalid log_history: %s", path)
            return None
        return state
    except json.JSONDecodeError as e:
        logger.warning("trainer_state.json parse error: %s — %s", path, e)
        return None
    except OSError as e:
        logger.warning("trainer_state.json read error: %s — %s", path, e)
        return None


def load_metrics(output_dir: str | Path):
    """
    Merge trainer_state.json log_history with metrics.jsonl (if present).
    Returns a list of dicts sorted by step.
    """
    output_dir = Path(output_dir)
    rows_by_step: dict = {}

    state_path = output_dir / "trainer_state.json"
    if state_path.exists():
        try:
            with open(state_path) as f:
                state = json.load(f)
            for entry in state.get("log_history", []):
                step = entry.get("step")
                if step is None:
                    continue
                rows_by_step.setdefault(step, {}).update(entry)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read trainer_state.json: %s", e)

    jsonl_path = output_dir / "metrics.jsonl"
    if jsonl_path.exists():
        try:
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    step = entry.get("step")
                    if step is None:
                        continue
                    rows_by_step.setdefault(step, {}).update(entry)
        except OSError as e:
            logger.warning("Could not read metrics.jsonl: %s", e)

    return [rows_by_step[s] for s in sorted(rows_by_step)]
