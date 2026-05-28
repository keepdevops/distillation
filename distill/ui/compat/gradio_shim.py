"""Gradio ↔ Reflex compatibility shim.

Allows both UIs to share the same backend logic without coupling.
Provides:
  - get_active_ui(): detect which UI is running
  - SharedEventBus: fire events that both UIs can subscribe to
  - state_bridge(): sync Reflex AppState ↔ Gradio session state
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── UI detection ───────────────────────────────────────────────────────────────

def get_active_ui() -> str:
    """Return 'gradio', 'reflex', or 'both'."""
    has_gr = _is_gradio_running()
    has_rx = _is_reflex_running()
    if has_gr and has_rx:
        return "both"
    if has_rx:
        return "reflex"
    return "gradio"


def _is_gradio_running() -> bool:
    try:
        import gradio as gr
        return gr.utils.get_space() is not None or True  # best-effort
    except Exception:
        return False


def _is_reflex_running() -> bool:
    try:
        import reflex  # noqa: F401  # type: ignore[import]
        return True
    except ImportError:
        return False


# ── Shared event bus ───────────────────────────────────────────────────────────

class SharedEventBus:
    """Publish/subscribe event bus shared between Gradio and Reflex components.

    Events are delivered synchronously to all subscribers in the publishing thread.
    For async Reflex state updates, use the RefreshState event handler from AppState.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def on(self, event: str, handler: Callable[[Any], None]) -> None:
        """Subscribe handler to event."""
        with self._lock:
            self._subscribers[event].append(handler)

    def off(self, event: str, handler: Callable | None = None) -> None:
        """Unsubscribe. If handler is None, remove all subscribers for event."""
        with self._lock:
            if handler is None:
                self._subscribers[event].clear()
            else:
                self._subscribers[event] = [
                    h for h in self._subscribers[event] if h is not handler
                ]

    def emit(self, event: str, data: Any = None) -> None:
        """Fire event, calling all subscribers."""
        with self._lock:
            handlers = list(self._subscribers.get(event, []))
        for h in handlers:
            try:
                h(data)
            except Exception as exc:
                logger.error("EventBus handler error (%s): %s", event, exc)

    def event_names(self) -> list[str]:
        with self._lock:
            return list(self._subscribers.keys())


# ── Module-level singleton ─────────────────────────────────────────────────────

_bus = SharedEventBus()


def get_event_bus() -> SharedEventBus:
    return _bus


# ── State bridge ───────────────────────────────────────────────────────────────

def state_bridge_emit_metrics(step: int, loss: float, phase: str = "") -> None:
    """Emit training metrics to all UIs via the event bus."""
    _bus.emit("training_metrics", {"step": step, "loss": loss, "phase": phase})


def state_bridge_emit_log(line: str) -> None:
    """Emit a log line to all UIs."""
    _bus.emit("training_log", line)


def state_bridge_emit_thermal(cpu: float, gpu: float, power: float) -> None:
    """Emit thermal readings to all UIs."""
    _bus.emit("thermal_update", {"cpu_temp": cpu, "gpu_temp": gpu, "total_power": power})


def wire_live_monitor_to_bus() -> None:
    """Connect LiveMonitor callbacks to the shared event bus."""
    try:
        from distill.ui.monitoring.live_monitor import get_monitor
        from distill.ui.state_manager import get_job

        def on_job_update(job: Any) -> None:
            _bus.emit("job_update", {
                "status": job.status,
                "step":   job.step,
                "loss":   job.loss,
                "phase":  job.phase,
            })

        get_monitor().on_update(on_job_update)
        logger.info("LiveMonitor wired to SharedEventBus")
    except Exception as exc:
        logger.warning("wire_live_monitor_to_bus failed: %s", exc)
