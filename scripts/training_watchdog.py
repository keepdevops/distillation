#!/usr/bin/env python3
"""
Autonomous training monitor: deterministic rules, no agent hallucination.
Watches trainer_state.json for loss plateau, optionally thermal; writes
suggestions + pause.flag. Designed for LaunchAgent (survives reboot).
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Default rules (overridable via --config)
DEFAULT_RULES = {
    "plateau": {
        "window": 3,           # last N loss deltas
        "max_delta": 0.001,    # if all deltas < this → plateau
        "min_points": 5,       # need at least N loss points
        "lr_scale": 0.8,      # multiply LR by this on plateau
    },
    "thermal": {
        "enabled": False,
        # M3: no raw °C via powermetrics. Use cpu_power_mw or thermal_pressure.
        "pause_if_over": 18000,   # mW (18W CPU) — M3 Max under heavy load
        "metric": "cpu_power_mw",  # "cpu_power_mw" | "thermal_pressure"
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


def check_thermal(rules):
    """
    Read thermal-related metric on Apple Silicon (M1/M2/M3).
    powermetrics requires sudo; use: sudo visudo → user ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
    """
    thermal_cfg = rules.get("thermal", {})
    metric = thermal_cfg.get("metric", "cpu_power_mw")

    try:
        if metric == "thermal_pressure":
            # Thermal pressure 0–100 (aggregate throttling signal)
            proc = subprocess.run(
                ["/usr/bin/powermetrics", "-s", "thermal", "-i", "1", "-n", "1"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode != 0:
                return None
            # Parse "Thermal pressure: X" or similar
            m = re.search(r"[Tt]hermal\s*(?:pressure|state)[:\s]+(\d+)", proc.stdout)
            if m:
                return int(m.group(1))
            return None

        # Default: CPU power (mW) — correlates with heat on M-series
        proc = subprocess.run(
            ["/usr/bin/powermetrics", "-n", "1", "--samplers", "cpu_power"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return None
        m = re.search(r"CPU Power:\s*(\d+)\s*mW", proc.stdout)
        if m:
            return int(m.group(1))
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("check_thermal failed: %s", e)
        return None


def run_tick(output_dir, rules, dry_run=False):
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

    # Thermal rule (M3: cpu_power_mw or thermal_pressure via powermetrics)
    if rules["thermal"]["enabled"]:
        value = check_thermal(rules)
        if value is not None and value >= rules["thermal"]["pause_if_over"]:
            pause_path = Path(output_dir) / "pause.flag"
            if not pause_path.exists():
                metric = rules["thermal"].get("metric", "cpu_power_mw")
                payload = {"reason": "thermal", "value": value, "metric": metric}
                pause_path.write_text(json.dumps(payload))
                current["action"] = "pause"
                current["reason"] = "thermal"
                current["thermal_value"] = value
                updated = True

    if updated and not dry_run:
        try:
            write_suggestions(output_dir, current, rules)
            logger.info("step=%s action=%s reason=%s", step, current.get("action"), current.get("reason"))
        except OSError as e:
            logger.error("write_suggestions failed: %s", e)


def main():
    p = argparse.ArgumentParser(description="Training watchdog: plateau/thermal rules")
    p.add_argument("output_dir", type=str, help="Training output dir (trainer_state.json)")
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
