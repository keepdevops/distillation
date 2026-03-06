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

    Checks every step after check_step (not just once). Baseline is the
    average of the first baseline_window logged losses — more robust than
    using only the very first loss (which may be abnormally high pre-warmup).

    Args:
        check_step:          Start checking for divergence after this many steps.
        divergence_threshold: Stop if loss > baseline * threshold.
        baseline_loss:       Override baseline (default: auto from first steps).
        baseline_window:     Number of initial losses to average for baseline.
    """

    def __init__(self, check_step: int = 20, divergence_threshold: float = 1.5,
                 baseline_loss: float = None, baseline_window: int = 3):
        self.check_step = check_step
        self.divergence_threshold = divergence_threshold
        self.baseline_loss = baseline_loss
        self.baseline_window = baseline_window
        self._initial_losses: list[float] = []
        self.stopped_early = False

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None or self.stopped_early:
            return control

        current_loss = logs.get("loss")
        if current_loss is None:
            return control

        # Accumulate initial losses to form a robust baseline
        if self.baseline_loss is None:
            self._initial_losses.append(current_loss)
            if len(self._initial_losses) >= self.baseline_window:
                self.baseline_loss = sum(self._initial_losses) / len(self._initial_losses)
                logger.info("Early stopping baseline: %.4f (avg of first %d losses)",
                            self.baseline_loss, self.baseline_window)

        if state.global_step < self.check_step or self.baseline_loss is None:
            return control

        threshold = self.baseline_loss * self.divergence_threshold
        if current_loss > threshold:
            logger.warning(
                "Early stopping at step %d: loss=%.4f > threshold=%.4f "
                "(baseline=%.4f × %.1f) — diverging, stopping to save time.",
                state.global_step, current_loss, threshold,
                self.baseline_loss, self.divergence_threshold,
            )
            self.stopped_early = True
            control.should_training_stop = True
            control.should_save = False

        return control
