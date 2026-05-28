"""Loading, success, and error state animation components.

Returns HTML strings with CSS animations for async feedback.
All animations are pure CSS — no external dependencies.
"""
from __future__ import annotations

import gradio as gr


# ── Spinner ────────────────────────────────────────────────────────────────────

_SPINNER_CSS = """
<style>
@keyframes distill-spin {
  to { transform: rotate(360deg); }
}
.distill-spinner {
  display: inline-block;
  width: 1.2rem; height: 1.2rem;
  border: 2.5px solid #1e293b;
  border-top-color: #6366f1;
  border-radius: 50%;
  animation: distill-spin 0.7s linear infinite;
  vertical-align: middle;
  margin-right: 0.4rem;
}
@keyframes distill-pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.4; }
}
.distill-pulse {
  animation: distill-pulse 1.5s ease-in-out infinite;
}
@keyframes distill-fadein {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0);   }
}
.distill-fadein {
  animation: distill-fadein 0.25s ease-out both;
}
@keyframes distill-progress {
  0%   { width: 0%; }
  100% { width: 100%; }
}
.distill-progress-bar {
  height: 3px; background: #6366f1; border-radius: 2px;
  animation: distill-progress 2s ease-in-out infinite alternate;
}
</style>
"""


def loading_html(message: str = "Loading...") -> str:
    """Return an HTML loading spinner with message."""
    return (
        f'{_SPINNER_CSS}'
        f'<div class="distill-fadein" style="display:flex;align-items:center;'
        f'color:#94a3b8;font-size:.875rem;padding:.5rem 0">'
        f'  <div class="distill-spinner"></div>'
        f'  <span class="distill-pulse">{message}</span>'
        f'</div>'
    )


def success_html(message: str = "Done!") -> str:
    """Return an HTML success banner with fade-in."""
    return (
        f'{_SPINNER_CSS}'
        f'<div class="distill-fadein banner-ok" style="'
        f'background:rgba(34,197,94,.12);border-left:3px solid #22c55e;'
        f'padding:.4rem .75rem;border-radius:0 .4rem .4rem 0;'
        f'color:#22c55e;font-size:.875rem;font-weight:600">'
        f'✅ {message}'
        f'</div>'
    )


def error_html(message: str = "An error occurred.") -> str:
    """Return an HTML error banner with fade-in."""
    return (
        f'{_SPINNER_CSS}'
        f'<div class="distill-fadein" style="'
        f'background:rgba(239,68,68,.12);border-left:3px solid #ef4444;'
        f'padding:.4rem .75rem;border-radius:0 .4rem .4rem 0;'
        f'color:#ef4444;font-size:.875rem">'
        f'❌ {message}'
        f'</div>'
    )


def progress_bar_html(pct: float, label: str = "") -> str:
    """Return an HTML progress bar (0–100)."""
    pct = max(0.0, min(100.0, pct))
    return (
        f'<div style="margin:.5rem 0">'
        f'  {"<div style=\\'font-size:.75rem;color:#94a3b8;margin-bottom:.2rem\\'>" + label + "</div>" if label else ""}'
        f'  <div style="background:#1e293b;border-radius:3px;height:6px;overflow:hidden">'
        f'    <div style="width:{pct:.0f}%;background:#6366f1;height:6px;'
        f'border-radius:3px;transition:width .4s ease"></div>'
        f'  </div>'
        f'  <div style="font-size:.7rem;color:#94a3b8;margin-top:.2rem;text-align:right">'
        f'    {pct:.0f}%</div>'
        f'</div>'
    )


def stage_badge_html(stage: str, status: str) -> str:
    """Return a coloured stage status badge."""
    colors = {
        "pending":   ("pill-gray",   "⬜"),
        "running":   ("pill-blue",   "🔄"),
        "completed": ("pill-green",  "✅"),
        "failed":    ("pill-red",    "❌"),
        "skipped":   ("pill-gray",   "⏭"),
    }
    css, icon = colors.get(status, ("pill-gray", "?"))
    return f'<span class="pill {css}">{icon} {stage}</span>'


# ── Gradio component helpers ───────────────────────────────────────────────────

def render_loading_state(message: str = "Processing...") -> gr.HTML:
    """Render a loading spinner HTML component."""
    return gr.HTML(value=loading_html(message))


def make_status_html(success: bool, message: str) -> str:
    """Return success or error HTML based on bool flag."""
    return success_html(message) if success else error_html(message)
