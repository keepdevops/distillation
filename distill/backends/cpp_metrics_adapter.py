"""Training metrics streaming adapter for the C++ MetricsHistory struct.

Bridges the LiveMonitor series (plain Python lists) into the C++
MetricsHistory / TrainingStepMetrics structs when available, then back
to dicts for Plotly consumption.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _try_cpp():
    try:
        import distill_cpp  # type: ignore[import]
        return distill_cpp.TrainingStepMetrics, distill_cpp.MetricsHistory
    except ImportError:
        return None, None


class MetricsAdapter:
    """Wraps LiveMonitor series in C++ MetricsHistory when available,
    otherwise falls back to pure-Python list operations."""

    def __init__(self) -> None:
        TrainingStepMetrics, MetricsHistory = _try_cpp()
        self._cpp_history = MetricsHistory() if MetricsHistory is not None else None
        self._py_steps: list[int] = []
        self._py_loss:  list[float] = []
        self._py_lr:    list[float] = []
        self._py_grad:  list[float] = []
        self._cpp_backed = self._cpp_history is not None

    def push(
        self,
        step: int,
        loss: float,
        lr: float = 0.0,
        grad_norm: float = 0.0,
        phase: str = "",
        backend: str = "",
        total_steps: int = 0,
        elapsed_sec: float = 0.0,
    ) -> None:
        """Push one training step into the history."""
        if self._cpp_history is not None:
            TrainingStepMetrics, _ = _try_cpp()
            if TrainingStepMetrics is not None:
                try:
                    m = TrainingStepMetrics()
                    m.step          = step
                    m.loss          = loss
                    m.learning_rate = lr
                    m.grad_norm     = grad_norm
                    m.phase         = phase
                    m.backend       = backend
                    m.total_steps   = total_steps
                    m.elapsed_sec   = elapsed_sec
                    self._cpp_history.push(m)
                    return
                except Exception as exc:
                    logger.warning("C++ MetricsHistory.push failed: %s", exc)

        self._py_steps.append(step)
        self._py_loss.append(loss)
        self._py_lr.append(lr)
        self._py_grad.append(grad_norm)

    def get_series(self) -> dict[str, list]:
        """Return {steps, loss, lr, grad, smoothed} for Plotly."""
        from distill.ui.components.log_parser import smooth

        if self._cpp_history is not None:
            try:
                steps = self._cpp_history.step_series()
                loss  = self._cpp_history.loss_series()
                return {
                    "steps":    steps,
                    "loss":     loss,
                    "lr":       self._py_lr,
                    "grad":     self._py_grad,
                    "smoothed": smooth(list(loss)),
                    "cpp_backed": True,
                }
            except Exception as exc:
                logger.warning("C++ series read failed: %s", exc)

        return {
            "steps":    list(self._py_steps),
            "loss":     list(self._py_loss),
            "lr":       list(self._py_lr),
            "grad":     list(self._py_grad),
            "smoothed": smooth(list(self._py_loss)),
            "cpp_backed": False,
        }

    def smoothed_loss(self, window: int = 10) -> float:
        """Return the smoothed loss using C++ method when available."""
        if self._cpp_history is not None:
            try:
                return self._cpp_history.smoothed_loss(window)
            except Exception:
                pass
        loss = self._py_loss
        if not loss:
            return 0.0
        tail = loss[-window:]
        return sum(tail) / len(tail)

    def clear(self) -> None:
        if self._cpp_history is not None:
            try:
                self._cpp_history.clear()
            except Exception:
                pass
        self._py_steps.clear(); self._py_loss.clear()
        self._py_lr.clear();    self._py_grad.clear()

    def __len__(self) -> int:
        if self._cpp_history is not None:
            try:
                return len(self._cpp_history)
            except Exception:
                pass
        return len(self._py_steps)

    @property
    def cpp_backed(self) -> bool:
        return self._cpp_backed


# Module-level singleton used by LiveMonitor
_adapter = MetricsAdapter()


def get_adapter() -> MetricsAdapter:
    return _adapter
