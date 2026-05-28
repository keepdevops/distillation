"""Typed pub/sub event bus — the central nervous system of the Wow Sausage Maker.

Replaces the ad-hoc SharedEventBus in distill/ui/compat/gradio_shim.py with a
fully-typed, topic-enum-driven system that all tabs, backends, and export formats
publish to and subscribe from.

Usage:
    from distill.ui.core.event_bus import bus, Topic

    bus.on(Topic.TRAINING_STEP, my_handler)
    bus.emit(Topic.TRAINING_STEP, {"step": 42, "loss": 1.23})
    bus.off(Topic.TRAINING_STEP, my_handler)
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Topic registry ─────────────────────────────────────────────────────────────

class Topic(Enum):
    # Training lifecycle
    TRAINING_START    = auto()
    TRAINING_STEP     = auto()   # {"step", "loss", "lr", "grad_norm", "phase"}
    TRAINING_COMPLETE = auto()   # {"output_dir", "metrics"}
    TRAINING_FAILED   = auto()   # {"error"}
    TRAINING_PAUSED   = auto()
    TRAINING_RESUMED  = auto()

    # Job management
    JOB_QUEUED     = auto()   # {"job_id", "config"}
    JOB_STARTED    = auto()   # {"job_id", "pid"}
    JOB_LOG_LINE   = auto()   # {"job_id", "line"}
    JOB_CANCELLED  = auto()   # {"job_id"}

    # Hardware / thermal
    THERMAL_UPDATE    = auto()  # {"cpu_temp", "gpu_temp", "total_power"}
    THERMAL_ALERT     = auto()  # {"metric", "value", "threshold"}
    HARDWARE_DETECTED = auto()  # {"device", "backend_hint", "ram_gb"}

    # LoRA / training metrics
    LORA_METRICS    = auto()   # {"step", "adapter_norm", "update_ratio"}
    OOM_DETECTED    = auto()   # {"batch_size", "retry_batch_size"}
    CHECKPOINT_SAVED = auto()  # {"path", "step"}

    # Export
    EXPORT_STARTED   = auto()  # {"format", "model_path"}
    EXPORT_COMPLETE  = auto()  # {"format", "output_path"}
    EXPORT_FAILED    = auto()  # {"format", "error"}

    # Evaluation
    EVAL_STARTED  = auto()
    EVAL_COMPLETE = auto()    # {"perplexity", "quality_score", "mt_bench"}

    # UI events
    TAB_CHANGED   = auto()    # {"tab": tab_key}
    CONFIG_LOADED = auto()    # {"preset_name", "config"}


# ── Event payload ──────────────────────────────────────────────────────────────

@dataclass
class Event:
    topic: Topic
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""          # who emitted (e.g. "live_monitor", "dpo_trainer")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


Handler = Callable[[Event], None]


# ── Event bus ──────────────────────────────────────────────────────────────────

class EventBus:
    """Thread-safe typed publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subs: dict[Topic, list[Handler]] = {t: [] for t in Topic}
        self._lock = threading.Lock()
        self._history: list[Event] = []
        self._max_history = 500

    def on(self, topic: Topic, handler: Handler) -> None:
        """Subscribe handler to topic."""
        with self._lock:
            if handler not in self._subs[topic]:
                self._subs[topic].append(handler)

    def off(self, topic: Topic, handler: Handler | None = None) -> None:
        """Unsubscribe handler (or all handlers) from topic."""
        with self._lock:
            if handler is None:
                self._subs[topic].clear()
            else:
                self._subs[topic] = [h for h in self._subs[topic] if h is not handler]

    def emit(self, topic: Topic, data: dict[str, Any] | None = None, source: str = "") -> None:
        """Publish an event to all subscribers. Non-blocking; errors are logged."""
        event = Event(topic=topic, data=data or {}, source=source)
        with self._lock:
            handlers = list(self._subs[topic])
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        for h in handlers:
            try:
                h(event)
            except Exception as exc:
                logger.error("EventBus handler error [%s]: %s", topic.name, exc)

    def emit_async(self, topic: Topic, data: dict[str, Any] | None = None,
                   source: str = "") -> None:
        """Publish in a daemon thread so callers never block."""
        t = threading.Thread(
            target=self.emit, args=(topic, data, source),
            daemon=True, name=f"bus-{topic.name}",
        )
        t.start()

    def recent(self, topic: Topic | None = None, n: int = 20) -> list[Event]:
        """Return the most recent events (optionally filtered by topic)."""
        with self._lock:
            history = list(self._history)
        if topic is not None:
            history = [e for e in history if e.topic == topic]
        return history[-n:]

    def subscriber_count(self, topic: Topic) -> int:
        with self._lock:
            return len(self._subs[topic])

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()


# ── Module-level singleton ─────────────────────────────────────────────────────

bus = EventBus()


def wire_live_monitor(monitor: Any) -> None:
    """Connect a LiveMonitor instance to the global bus."""
    def on_job_update(job: Any) -> None:
        bus.emit(Topic.TRAINING_STEP, {
            "step":  job.step,
            "loss":  job.loss,
            "phase": job.phase,
        }, source="live_monitor")

    monitor.on_update(on_job_update)
    logger.info("LiveMonitor wired to EventBus")
