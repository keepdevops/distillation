"""Keyboard shortcut handler for the Gradio UI.

Injects a small JavaScript snippet that maps key chords to Gradio button
clicks. Since Gradio doesn't expose a native keybinding API, we use
gr.HTML with embedded <script> tags.

Available shortcuts (configurable):
  Ctrl+Enter   — Start training (primary action button on current tab)
  Ctrl+S       — Save current config to session
  Ctrl+R       — Refresh hardware / thermal status
  Escape       — Cancel / stop running job
  Ctrl+Shift+E — Open Export tab
  Ctrl+Shift+L — Open Logs tab
"""
from __future__ import annotations

import gradio as gr

# ── Default shortcut map ──────────────────────────────────────────────────────
# key combo → CSS selector of the button/element to click

DEFAULT_SHORTCUTS: dict[str, str] = {
    "ctrl+enter":   "[data-testid='start-btn']",
    "ctrl+s":       "[data-testid='save-config-btn']",
    "ctrl+r":       "[data-testid='refresh-btn']",
    "escape":       "[data-testid='cancel-btn']",
}

_SHORTCUT_JS_TEMPLATE = """
<script>
(function() {{
  const shortcuts = {shortcuts_json};

  document.addEventListener('keydown', function(e) {{
    const ctrl  = e.ctrlKey || e.metaKey;
    const shift = e.shiftKey;
    const key   = e.key.toLowerCase();

    let combo = '';
    if (ctrl  && shift) combo = 'ctrl+shift+' + key;
    else if (ctrl)      combo = 'ctrl+'        + key;
    else if (shift)     combo = 'shift+'       + key;
    else                combo = key;

    const selector = shortcuts[combo];
    if (!selector) return;

    // Don't fire in text inputs
    const active = document.activeElement;
    if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) {{
      if (combo !== 'escape') return;
    }}

    const el = document.querySelector(selector);
    if (el) {{
      e.preventDefault();
      el.click();
      console.debug('[distill-ui] Shortcut', combo, '→', selector);
    }}
  }});

  // Show shortcut hints on first load
  if (!sessionStorage.getItem('distill_shortcuts_shown')) {{
    sessionStorage.setItem('distill_shortcuts_shown', '1');
    console.info('[distill-ui] Keyboard shortcuts:', Object.keys(shortcuts).join(', '));
  }}
}})();
</script>
"""

_HELP_MD = """
| Shortcut | Action |
|---|---|
| `Ctrl+Enter` | Start training (primary button) |
| `Ctrl+S` | Save current config |
| `Ctrl+R` | Refresh hardware status |
| `Esc` | Cancel / stop running job |
"""


def build_shortcut_script(shortcuts: dict[str, str] | None = None) -> str:
    """Return the JS snippet that registers keyboard shortcuts."""
    import json
    mapping = {**DEFAULT_SHORTCUTS, **(shortcuts or {})}
    return _SHORTCUT_JS_TEMPLATE.format(shortcuts_json=json.dumps(mapping))


def render_keyboard_shortcuts(shortcuts: dict[str, str] | None = None) -> None:
    """Inject keyboard shortcut JS and a help accordion into the UI."""
    script_html = build_shortcut_script(shortcuts)
    gr.HTML(value=script_html)

    with gr.Accordion("⌨ Keyboard Shortcuts", open=False):
        gr.Markdown(_HELP_MD)


def shortcut_help_markdown() -> str:
    return _HELP_MD
