"""
Trainer callbacks for watchdog integration.
Add to distill_minillm.py / distill_forward.py for pause.flag support.
"""

import json
import logging
from pathlib import Path

from transformers import TrainerCallback

log = logging.getLogger(__name__)


class PauseFlagCallback(TrainerCallback):
    """
    Checks for pause.flag in output_dir; if present, saves and exits gracefully.
    Used with training_watchdog.py for thermal/plateau interventions.
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def on_step_end(self, args, state, control, **kwargs):
        flag_path = self.output_dir / "pause.flag"
        if not flag_path.exists():
            return control
        try:
            with open(flag_path) as f:
                info = json.load(f)
            reason = info.get("reason", "unknown")
        except json.JSONDecodeError as e:
            log.warning("pause.flag parse error: %s", e)
            reason = "pause.flag"
        except OSError as e:
            log.warning("pause.flag read error: %s", e)
            reason = "pause.flag"
        log.info("Detected %s (reason=%s). Saving and exiting.", flag_path, reason)
        control.should_training_stop = True
        return control


class MetricsCallback(TrainerCallback):
    """
    Streams all Trainer log events to {output_dir}/metrics.jsonl in real-time.
    One JSON object per line; captures all keys (loss, grad_norm, KL, etc.).
    """

    def __init__(self, output_dir):
        self.path = Path(output_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
