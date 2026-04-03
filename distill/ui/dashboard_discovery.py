"""
Dashboard discovery helpers: locate training run directories.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_run_dirs(runs_dir):
    """Find directories containing trainer_state.json."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    found = []
    try:
        for d in root.iterdir():
            if d.is_dir() and (d / "trainer_state.json").exists():
                found.append(str(d))
        # Also check runs_dir itself
        if (root / "trainer_state.json").exists():
            found.insert(0, str(root))
    except PermissionError as exc:
        logger.error("Permission denied scanning %s: %s", runs_dir, exc)
    return sorted(set(found))


def find_pipeline_dirs(runs_dir):
    """
    Find directories suitable for the pipeline view.
    A directory qualifies if it looks like a distillation output:
    - has trainer_state.json at root or inside a checkpoint subdir, OR
    - has *.gguf files AND (config.json or training_args.bin)
    Returns absolute paths.
    """
    root = Path(runs_dir).resolve()
    if not root.exists():
        return []
    found = set()
    try:
        candidates = [root] + [d for d in root.iterdir() if d.is_dir()]
    except PermissionError as exc:
        logger.error("Permission denied listing %s: %s", runs_dir, exc)
        return []

    for d in candidates:
        if (d / "trainer_state.json").exists():
            found.add(str(d))
            continue
        # trainer_state.json inside a checkpoint subdir
        try:
            if any(
                (sub / "trainer_state.json").exists()
                for sub in d.iterdir()
                if sub.is_dir()
            ):
                found.add(str(d))
                continue
        except PermissionError:
            continue
        # Live run: metrics.jsonl exists (trainer_state.json not yet written)
        if (d / "metrics.jsonl").exists():
            found.add(str(d))
            continue
        # GGUF files alongside a config.json or training_args.bin
        if list(d.glob("*.gguf")) and (
            (d / "config.json").exists() or (d / "training_args.bin").exists()
        ):
            found.add(str(d))

    return sorted(found)
