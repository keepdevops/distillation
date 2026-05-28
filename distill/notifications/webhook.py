"""Webhook notifications — POST to Slack/Discord on run events.

Events: run_complete, run_failed, thermal_alert, checkpoint_saved.
URLs are read from environment variables or webhook_config.py — never hardcoded.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WebhookEvent:
    event: str          # run_complete / run_failed / thermal_alert / checkpoint_saved
    title: str
    body: str
    data: dict[str, Any] = None  # type: ignore[assignment]
    color: int = 0x6366f1        # default indigo

    def __post_init__(self):
        if self.data is None:
            self.data = {}

    def to_slack_payload(self) -> dict:
        icon = {
            "run_complete":      "✅",
            "run_failed":        "❌",
            "thermal_alert":     "🌡",
            "checkpoint_saved":  "💾",
        }.get(self.event, "ℹ")
        return {
            "text": f"{icon} *{self.title}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"{icon} *{self.title}*\n{self.body}"},
                }
            ],
        }

    def to_discord_payload(self) -> dict:
        return {
            "embeds": [{
                "title":       self.title,
                "description": self.body,
                "color":       self.color,
                "footer":      {"text": f"distill · event: {self.event}"},
            }]
        }


def _post(url: str, payload: dict, dry_run: bool = False) -> bool:
    """POST JSON payload to url. Returns True on success."""
    if dry_run:
        logger.info("[DRY RUN] Would POST to %s: %s", url[:60], json.dumps(payload)[:200])
        return True
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            ok = resp.status < 300
            if not ok:
                logger.warning("Webhook POST returned %d", resp.status)
            return ok
    except Exception as exc:
        logger.error("Webhook POST failed: %s", exc)
        return False


def send(
    event: str,
    title: str,
    body: str,
    data: dict | None = None,
    dry_run: bool = False,
    blocking: bool = False,
) -> None:
    """Fire a webhook event. Non-blocking by default (background thread).

    Args:
        event:    Event key (run_complete / run_failed / thermal_alert / checkpoint_saved).
        title:    Short title for the notification.
        body:     Longer description / metrics summary.
        data:     Optional extra dict attached to the event.
        dry_run:  Log instead of actually POSTing.
        blocking: Wait for the POST to complete before returning.
    """
    from distill.notifications.webhook_config import get_urls
    urls = get_urls(event)
    if not urls and not dry_run:
        return  # no webhooks configured — silently skip

    ev = WebhookEvent(event=event, title=title, body=body, data=data or {})

    def _fire():
        for url_entry in urls:
            platform = url_entry.get("platform", "slack")
            url      = url_entry.get("url", "")
            if not url:
                continue
            payload = (
                ev.to_discord_payload() if platform == "discord"
                else ev.to_slack_payload()
            )
            _post(url, payload, dry_run=dry_run)

    if blocking:
        _fire()
    else:
        t = threading.Thread(target=_fire, daemon=True, name="WebhookFire")
        t.start()


# ── Convenience helpers ────────────────────────────────────────────────────────

def notify_run_complete(
    model: str, backend: str, loss: float, quality: float,
    elapsed: str, dry_run: bool = False,
) -> None:
    send(
        event="run_complete",
        title=f"Run Complete — {model}",
        body=(
            f"Backend: `{backend}` · Loss: `{loss:.4f}` · "
            f"Quality: `{quality:.3f}` · Time: `{elapsed}`"
        ),
        data={"model": model, "backend": backend, "loss": loss, "quality": quality},
        dry_run=dry_run,
    )


def notify_thermal_alert(cpu: float, gpu: float, threshold: float, dry_run: bool = False) -> None:
    send(
        event="thermal_alert",
        title="⚠ Thermal Alert",
        body=f"CPU: `{cpu:.0f}°C` · GPU: `{gpu:.0f}°C` · Threshold: `{threshold:.0f}°C`",
        data={"cpu_temp": cpu, "gpu_temp": gpu, "threshold": threshold},
        dry_run=dry_run,
        color=0xef4444,
    )
