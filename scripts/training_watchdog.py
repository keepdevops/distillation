#!/usr/bin/env python3
"""
Autonomous training monitor: ML-specific progress monitoring.

Watches trainer_state.json for loss plateau; writes suggestions + pause.flag.
Designed for LaunchAgent (survives reboot).

Note: For thermal control, use the standalone thermal_agent.py instead.
This script focuses on ML-specific monitoring (plateau detection).
"""

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Default rules (overridable via --config)
# Note: Thermal control moved to standalone thermal_agent.py
DEFAULT_RULES = {
    "plateau": {
        "window": 3,           # last N loss deltas
        "max_delta": 0.001,    # if all deltas < this = plateau
        "min_points": 5,       # need at least N loss points
        "lr_scale": 0.8,      # multiply LR by this on plateau
    },
    "divergence": {
        "window": 3,           # average of last N losses for "current"
        "threshold": 1.5,      # pause if recent avg > baseline avg × this
        "baseline_window": 5,  # average of first N losses for baseline
        "min_points": 8,       # need at least N points before checking
    },
    "validator": {
        "backup_before_write": True,
        "max_lr_scale": 0.5,   # never suggest LR below 50% of original
    },
}


def load_trainer_state(output_dir):
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


def get_recent_losses(state, n=10):
    """Extract last n training loss values from log_history."""
    log = state.get("log_history", [])
    losses = []
    for e in reversed(log):
        if "loss" in e:
            losses.append(e["loss"])
            if len(losses) >= n:
                break
    return list(reversed(losses))


def detect_plateau(losses, rules):
    """If last N deltas are all < max_delta, return True."""
    w = rules["plateau"]["window"]
    max_d = rules["plateau"]["max_delta"]
    min_p = rules["plateau"]["min_points"]
    if len(losses) < min_p or len(losses) < w + 1:
        return False
    recent = losses[-(w + 1):]
    deltas = [abs(recent[i + 1] - recent[i]) for i in range(len(recent) - 1)]
    return all(d < max_d for d in deltas)


def detect_divergence(losses, rules):
    """Return (is_diverging, recent_avg, baseline_avg).

    Diverging = recent average loss > early baseline average × threshold.
    """
    div = rules.get("divergence", {})
    window = div.get("window", 3)
    threshold = div.get("threshold", 1.5)
    baseline_window = div.get("baseline_window", 5)
    min_points = div.get("min_points", 8)
    if len(losses) < min_points:
        return False, None, None
    n_base = min(baseline_window, len(losses))
    baseline = sum(losses[:n_base]) / n_base
    n_recent = min(window, len(losses))
    recent_avg = sum(losses[-n_recent:]) / n_recent
    return recent_avg > baseline * threshold, recent_avg, baseline


def read_current_suggestions(output_dir):
    path = Path(output_dir) / "watchdog_suggestions.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.warning("watchdog_suggestions.json parse error: %s — %s", path, e)
        return {}
    except OSError as e:
        logger.warning("watchdog_suggestions.json read error: %s — %s", path, e)
        return {}


def write_suggestions(output_dir, suggestions, rules):
    """Atomic write with backup and validator."""
    p = Path(output_dir)
    target = p / "watchdog_suggestions.json"
    backup = p / "watchdog_suggestions.json.bak"
    if rules["validator"]["backup_before_write"] and target.exists():
        shutil.copy(target, backup)
    # Validator: clamp lr_scale
    if "next_lr_scale" in suggestions:
        suggestions["next_lr_scale"] = max(
            suggestions["next_lr_scale"],
            rules["validator"]["max_lr_scale"],
        )
    tmp = target.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(suggestions, f, indent=2)
    tmp.rename(target)


def run_tick(output_dir, rules, dry_run=False):
    """
    Monitor training progress and write suggestions.

    Note: Thermal monitoring removed - use thermal_agent.py for system-wide thermal control.
    """
    state = load_trainer_state(output_dir)
    if not state:
        return
    log = state.get("log_history", [])
    if not log:
        return
    losses = get_recent_losses(state, n=20)
    if len(losses) < rules["plateau"]["min_points"]:
        return
    current = read_current_suggestions(output_dir)
    step = next((e.get("step") for e in reversed(log) if "step" in e), 0)
    updated = False

    # Plateau rule
    if detect_plateau(losses, rules):
        lr_scale = rules["plateau"]["lr_scale"]
        prev_scale = current.get("next_lr_scale", 1.0)
        new_scale = prev_scale * lr_scale
        if new_scale != prev_scale:
            current["action"] = "lr_scale"
            current["next_lr_scale"] = new_scale
            current["reason"] = "plateau"
            current["at_step"] = step
            current["last_losses"] = losses[-5:]
            updated = True

    # Divergence rule — write pause.flag to stop training
    diverged, recent_avg, baseline_avg = detect_divergence(losses, rules)
    if diverged:
        pause_flag = Path(output_dir) / "pause.flag"
        if not pause_flag.exists():
            try:
                pause_flag.touch()
            except OSError as e:
                logger.error("Failed to write pause.flag: %s", e)
        threshold = rules.get("divergence", {}).get("threshold", 1.5)
        logger.warning(
            "step=%s DIVERGENCE: recent_avg=%.4f > baseline=%.4f × %.1f — wrote pause.flag",
            step, recent_avg, baseline_avg, threshold,
        )
        current["action"] = "pause"
        current["reason"] = "divergence"
        current["at_step"] = step
        current["recent_avg_loss"] = round(recent_avg, 4)
        current["baseline_avg_loss"] = round(baseline_avg, 4)
        current["last_losses"] = losses[-5:]
        updated = True

    if updated and not dry_run:
        try:
            write_suggestions(output_dir, current, rules)
            logger.info("step=%s action=%s reason=%s", step, current.get("action"), current.get("reason"))
        except OSError as e:
            logger.error("write_suggestions failed: %s", e)


def main():
    p = argparse.ArgumentParser(
        description="Training watchdog: ML-specific progress monitoring (plateau detection). "
                    "For thermal control, use thermal_agent.py instead."
    )
    p.add_argument("output_dir", type=str, nargs="?", default="./distilled-minillm", help="Training output dir (trainer_state.json)")
    p.add_argument("--interval", type=int, default=60, help="Poll interval seconds")
    p.add_argument("--config", type=str, default=None, help="JSON rules file")
    p.add_argument("--once", action="store_true", help="Run one tick and exit")
    p.add_argument("--dry-run", action="store_true", help="Log only, do not write")
    args = p.parse_args()
    if args.interval <= 0:
        logger.warning("interval must be > 0, using 60")
        args.interval = 60
    rules = DEFAULT_RULES.copy()
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.warning("Config file not found: %s — using defaults", args.config)
        else:
            try:
                with open(config_path) as f:
                    rules = {**rules, **json.load(f)}
                logger.info("Loaded config from %s", config_path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Config load failed: %s — %s", config_path, e)
    output_dir = os.path.abspath(args.output_dir)
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    if args.once:
        run_tick(output_dir, rules, args.dry_run)
        return
    logger.info("Monitoring %s every %ss", output_dir, args.interval)
    while True:
        run_tick(output_dir, rules, args.dry_run)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
