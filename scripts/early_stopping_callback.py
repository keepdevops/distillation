#!/usr/bin/env python3
"""
Early stopping callback for diverging trials.
Stops training early if loss is significantly higher than baseline after N steps.
"""

import logging
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainerCallback):
    """
    Stop training early if loss diverges significantly from baseline.

    Args:
        check_step: Step at which to check for divergence (default: 20)
        divergence_threshold: Stop if loss > baseline * threshold (default: 1.5)
        baseline_loss: Baseline loss to compare against (default: None, auto-detect from first steps)
    """

    def __init__(self, check_step: int = 20, divergence_threshold: float = 1.5, baseline_loss: float = None):
        self.check_step = check_step
        self.divergence_threshold = divergence_threshold
        self.baseline_loss = baseline_loss
        self.first_loss = None
        self.stopped_early = False

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Check loss at intervals and stop if diverging."""
        if logs is None or self.stopped_early:
            return control

        current_step = state.global_step
        current_loss = logs.get("loss")

        if current_loss is None:
            return control

        # Record first loss as baseline if not provided
        if self.first_loss is None:
            self.first_loss = current_loss
            if self.baseline_loss is None:
                self.baseline_loss = current_loss
                logger.info(f"Early stopping baseline set: loss={self.baseline_loss:.4f}")

        # Check at specified step
        if current_step >= self.check_step and not self.stopped_early:
            threshold_loss = self.baseline_loss * self.divergence_threshold

            if current_loss > threshold_loss:
                logger.warning(
                    f"Early stopping triggered at step {current_step}: "
                    f"loss={current_loss:.4f} > threshold={threshold_loss:.4f} "
                    f"(baseline={self.baseline_loss:.4f} × {self.divergence_threshold})"
                )
                logger.warning("Trial is diverging, stopping early to save time")
                self.stopped_early = True
                control.should_training_stop = True
                control.should_save = False  # Don't save diverged checkpoint
            else:
                logger.info(
                    f"Early stopping check at step {current_step}: "
                    f"loss={current_loss:.4f} <= threshold={threshold_loss:.4f} (OK, continuing)"
                )

        return control
