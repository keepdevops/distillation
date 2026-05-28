"""Air-gapped environment indicator and mode controller.

Detects whether the system is air-gapped (no internet access) and provides
a UI banner + helper to enable offline-only operation across all components.
"""
from __future__ import annotations

import logging
import os
import socket
from functools import lru_cache
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

_PROBE_HOST = "huggingface.co"
_PROBE_PORT = 443
_PROBE_TIMEOUT = 2.0


@lru_cache(maxsize=1)
def _detect_internet(timeout: float = _PROBE_TIMEOUT) -> bool:
    """Return True if internet is reachable (cached for session lifetime)."""
    # Honour explicit env override
    if os.environ.get("DISTILL_AIRGAP", "").lower() in ("1", "true", "yes"):
        return False
    try:
        socket.setdefaulttimeout(timeout)
        with socket.create_connection((_PROBE_HOST, _PROBE_PORT)):
            return True
    except OSError:
        return False


def is_airgap() -> bool:
    """Return True when running in air-gapped (offline) mode."""
    return not _detect_internet()


def airgap_banner_html() -> str:
    """Return an HTML banner appropriate to current connectivity status."""
    if is_airgap():
        return (
            '<div class="banner-warning" style="margin-bottom:.75rem">'
            '  ✈ <b>Air-gapped mode</b> — no internet connection detected. '
            '  Only locally cached models and datasets are available. '
            '  Run <code>distill-setup-airgap</code> to pre-cache dependencies.'
            '</div>'
        )
    return (
        '<div style="background:rgba(34,197,94,.08);border-left:3px solid #22c55e;'
        'padding:.4rem .75rem;border-radius:0 .4rem .4rem 0;'
        'font-size:.8rem;color:#94a3b8;margin-bottom:.75rem">'
        '  🌐 Online — HuggingFace Hub reachable'
        '</div>'
    )


def airgap_env_vars() -> dict[str, str]:
    """Return environment variable overrides for offline operation."""
    if not is_airgap():
        return {}
    return {
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }


def apply_airgap_env() -> None:
    """Set offline env vars in the current process if air-gapped."""
    for k, v in airgap_env_vars().items():
        os.environ.setdefault(k, v)
    if is_airgap():
        logger.info("Air-gapped mode: set HF offline env vars")


def render_airgap_banner() -> gr.HTML:
    """Render the air-gap status banner inside the current gr.Blocks context."""
    return gr.HTML(value=airgap_banner_html)


def cached_model_list() -> list[str]:
    """Return locally cached HF model IDs from the hub cache."""
    from pathlib import Path
    cache_dir = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub"
    if not cache_dir.exists():
        return []
    models: list[str] = []
    for p in cache_dir.glob("models--*"):
        name = p.name.replace("models--", "").replace("--", "/")
        models.append(name)
    return sorted(models)


def connectivity_status() -> dict[str, Any]:
    """Return a status dict for the Hardware tab and settings pages."""
    online = not is_airgap()
    return {
        "online":      online,
        "airgap":      not online,
        "hf_offline":  os.environ.get("HF_HUB_OFFLINE", "0") == "1",
        "cached_models": len(cached_model_list()),
        "probe_host":  _PROBE_HOST,
    }
