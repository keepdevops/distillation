"""Tab registry — maps string keys to zero-arg tab builder callables.

Each builder is imported lazily so a broken tab doesn't prevent the app
from starting. Existing tabs that require args are wrapped in closures
that supply discovered models/datasets at import time.
"""
from __future__ import annotations

import importlib
import logging
from typing import Callable

import gradio as gr

logger = logging.getLogger(__name__)


def _discovery():
    """Lazy-load discovery results (cached after first call)."""
    if not hasattr(_discovery, "_cache"):
        try:
            from distill.launch_ui.discovery import (
                discover_teachers, discover_students,
                discover_datasets, discover_output_dirs,
            )
            from distill.infra.paths import project_dir
            teachers = discover_teachers()
            students = discover_students()
            datasets = discover_datasets()
            out_dirs = discover_output_dirs()
            project = project_dir()
            defaults = {
                "default_teacher": "Qwen/Qwen2-1.5B-Instruct",
                "default_student": students[0] if students else "Qwen/Qwen2-0.5B-Instruct",
                "default_dataset": "yahma/alpaca-cleaned",
                "default_out":     str(project / "distilled-minillm"),
            }
            _discovery._cache = {
                "teachers": teachers, "students": students,
                "datasets": datasets, "out_dirs": out_dirs,
                "defaults": defaults,
            }
        except Exception as exc:
            logger.warning("discovery failed: %s", exc)
            _discovery._cache = {
                "teachers": [], "students": [], "datasets": [],
                "out_dirs": [], "defaults": {
                    "default_teacher": "", "default_student": "",
                    "default_dataset": "", "default_out": "outputs",
                },
            }
    return _discovery._cache


def _wrap_data_prep():
    d = _discovery()
    from distill.launch_ui.tabs.tab_data_prep import build_tab_data_prep
    build_tab_data_prep(d["teachers"], d["datasets"])


def _wrap_sft():
    d = _discovery()
    from distill.launch_ui.tabs.tab_configure import build_tab_configure
    build_tab_configure(d["teachers"], d["students"], d["datasets"],
                        d["out_dirs"], d["defaults"])


def _wrap_distillation():
    d = _discovery()
    from distill.launch_ui.tabs.tab_expert import build_tab_expert
    build_tab_expert(d["students"])


def _wrap_eval():
    d = _discovery()
    from distill.launch_ui.tabs.tab_eval import build_tab_eval
    defs = d["defaults"]
    build_tab_eval(
        d["teachers"], d["students"], d["datasets"], d["out_dirs"],
        defs["default_teacher"], defs["default_student"],
        defs["default_dataset"], defs["default_out"],
    )


def _wrap_domain():
    d = _discovery()
    from distill.launch_ui.tabs.tab_domain import build_tab_domain
    build_tab_domain(d["teachers"])


def _wrap_generate():
    from distill.eval.gradio_ui_tabs import build_generate_tab
    build_generate_tab()


def _wrap_export():
    d = _discovery()
    from distill.eval.gradio_ui_tab_export import build_export_tab
    paths = d.get("out_dirs", [])
    build_export_tab(paths[0] if paths else ".")


# ── Registry: key → zero-arg callable ────────────────────────────────────────
_BUILTIN: dict[str, Callable] = {
    # New tabs (Sprint 1+)
    "hardware":     None,   # resolved below
    "swarm_export": None,
    "alignment":    None,
    "experiments":  None,
    # Wrapped existing tabs
    "data_prep":    _wrap_data_prep,
    "sft":          _wrap_sft,
    "distillation": _wrap_distillation,
    "eval":         _wrap_eval,
    "full_auto":    _wrap_domain,
    "generate":     _wrap_generate,
    "export":       _wrap_export,
    "logs":         None,
    "help":         None,
}

# Dynamic import registry for new tabs (module_path, fn_name)
_DYNAMIC: dict[str, tuple[str, str]] = {
    "hardware":       ("distill.ui.tabs.hardware",          "build_tab"),
    "swarm_export":   ("distill.ui.tabs.swarm_export",      "build_tab"),
    "alignment":      ("distill.ui.tabs.alignment",         "build_tab"),
    "experiments":    ("distill.ui.tabs.experiments",       "build_tab"),
    "training_live":   ("distill.ui.tabs.training_live",    "build_tab"),
    "eval_comparison": ("distill.ui.tabs.eval_comparison",  "build_tab"),
    "quantize_export":   ("distill.ui.tabs.quantize_export",   "build_tab"),
    "configure_backend": ("distill.ui.tabs.configure_backend", "build_tab"),
    "full_auto_gantt":  ("distill.ui.tabs.full_auto_gantt",  "build_tab"),
    "logs":           ("distill.launch_ui.tabs.tab_logs",   "build_tab_logs"),
    "help":           ("distill.launch_ui.tabs.tab_help",   "build_tab_help"),
}

_CACHE: dict[str, Callable] = {}

# Cache resolved callables to avoid repeated importlib calls
_CACHE: dict[str, Callable] = {}


def build_tab(key: str) -> None:
    """Look up and invoke the tab builder for *key* within the current gr.Blocks context."""
    builder = _resolve(key)
    if builder is None:
        gr.Markdown(f"⚠ Tab `{key}` is not registered. Check `tab_registry.py`.")
        return
    try:
        builder()
    except Exception as exc:
        logger.error("Tab '%s' builder raised: %s", key, exc, exc_info=True)
        gr.Markdown(
            f"**Tab `{key}` failed to render.**\n\n```\n{exc}\n```\n\n"
            "Check the terminal for a full traceback."
        )


def _resolve(key: str) -> Callable | None:
    """Return the builder callable for *key*, or None if unavailable."""
    if key in _CACHE:
        return _CACHE[key]

    # Check builtin wrappers first
    if key in _BUILTIN and _BUILTIN[key] is not None:
        _CACHE[key] = _BUILTIN[key]
        return _BUILTIN[key]

    # Fall back to dynamic import
    if key not in _DYNAMIC:
        logger.warning("Unknown tab key: '%s'", key)
        return None

    module_path, fn_name = _DYNAMIC[key]
    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, fn_name)
        _CACHE[key] = fn
        return fn
    except ImportError as exc:
        logger.warning("Tab '%s' import failed (%s): %s", key, module_path, exc)
    except AttributeError as exc:
        logger.warning("Tab '%s' missing '%s' in %s: %s", key, fn_name, module_path, exc)
    return None


def register_tab(key: str, module_path: str, fn_name: str = "build_tab") -> None:
    """Register a custom tab at runtime (for plugins / extensions)."""
    _DYNAMIC[key] = (module_path, fn_name)
    _CACHE.pop(key, None)
    logger.info("Registered custom tab: '%s' -> %s.%s", key, module_path, fn_name)
