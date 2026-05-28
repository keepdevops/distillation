"""Webhook URL configuration — environment-variable backed, never hardcoded.

Environment variables:
  DISTILL_WEBHOOK_SLACK    — Slack incoming webhook URL
  DISTILL_WEBHOOK_DISCORD  — Discord webhook URL
  DISTILL_WEBHOOK_EVENTS   — Comma-separated events to fire on (default: all)

All events: run_complete, run_failed, thermal_alert, checkpoint_saved
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ALL_EVENTS = {"run_complete", "run_failed", "thermal_alert", "checkpoint_saved"}

# Runtime-registered webhooks (set via UI settings tab)
_runtime_hooks: list[dict] = []


def get_urls(event: str) -> list[dict]:
    """Return list of {platform, url} dicts for the given event."""
    allowed = _get_allowed_events()
    if event not in allowed:
        return []

    hooks: list[dict] = []

    # Environment-variable sources
    slack_url = os.environ.get("DISTILL_WEBHOOK_SLACK", "").strip()
    if slack_url:
        hooks.append({"platform": "slack", "url": slack_url})

    discord_url = os.environ.get("DISTILL_WEBHOOK_DISCORD", "").strip()
    if discord_url:
        hooks.append({"platform": "discord", "url": discord_url})

    # Runtime-registered hooks (added via UI)
    hooks.extend(h for h in _runtime_hooks if h.get("url"))

    return hooks


def register(platform: str, url: str, events: list[str] | None = None) -> None:
    """Register a webhook URL at runtime (e.g., from the Settings UI)."""
    if not url.startswith("https://"):
        logger.warning("Webhook URL should start with https://")
    entry = {"platform": platform, "url": url, "events": events or list(_ALL_EVENTS)}
    _runtime_hooks.append(entry)
    logger.info("Registered %s webhook (events=%s)", platform, entry["events"])


def unregister_all() -> None:
    _runtime_hooks.clear()


def is_configured() -> bool:
    """Return True if at least one webhook is configured."""
    return bool(
        os.environ.get("DISTILL_WEBHOOK_SLACK")
        or os.environ.get("DISTILL_WEBHOOK_DISCORD")
        or _runtime_hooks
    )


def status_markdown() -> str:
    """Return a markdown summary of configured webhooks for display in the UI."""
    lines = ["| Platform | Source | Events |", "|---|---|---|"]
    slack = os.environ.get("DISTILL_WEBHOOK_SLACK", "")
    discord = os.environ.get("DISTILL_WEBHOOK_DISCORD", "")
    if slack:
        lines.append("| Slack | env: DISTILL_WEBHOOK_SLACK | all |")
    if discord:
        lines.append("| Discord | env: DISTILL_WEBHOOK_DISCORD | all |")
    for h in _runtime_hooks:
        lines.append(f"| {h['platform']} | runtime | {', '.join(h.get('events', ['all']))} |")
    if len(lines) == 2:
        return "*No webhooks configured. Set DISTILL_WEBHOOK_SLACK or DISTILL_WEBHOOK_DISCORD.*"
    return "\n".join(lines)


def _get_allowed_events() -> set[str]:
    raw = os.environ.get("DISTILL_WEBHOOK_EVENTS", "").strip()
    if not raw:
        return _ALL_EVENTS
    return {e.strip() for e in raw.split(",") if e.strip()} & _ALL_EVENTS
