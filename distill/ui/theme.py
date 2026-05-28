"""Shared Gradio theme and CSS design tokens for the Wow Sausage Maker UI."""
from __future__ import annotations

import gradio as gr

# ── Design tokens ────────────────────────────────────────────────────────────
COLOR_PRIMARY = "#6366f1"       # indigo-500
COLOR_PRIMARY_DARK = "#4f46e5"  # indigo-600
COLOR_SUCCESS = "#22c55e"       # green-500
COLOR_WARNING = "#f59e0b"       # amber-500
COLOR_ERROR = "#ef4444"         # red-500
COLOR_SURFACE = "#1e1e2e"       # dark surface
COLOR_SURFACE_ALT = "#2a2a3e"   # slightly lighter surface
COLOR_TEXT = "#e2e8f0"          # slate-200
COLOR_TEXT_MUTED = "#94a3b8"    # slate-400
COLOR_BORDER = "#374151"        # gray-700

SPACING_SM = "0.5rem"
SPACING_MD = "1rem"
SPACING_LG = "1.5rem"
RADIUS = "0.5rem"
FONT_MONO = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace"

# ── Gradio Theme ─────────────────────────────────────────────────────────────

def build_theme() -> gr.Theme:
    """Return a dark, professional Gradio theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace"],
    ).set(
        body_background_fill=COLOR_SURFACE,
        body_background_fill_dark=COLOR_SURFACE,
        body_text_color=COLOR_TEXT,
        body_text_color_dark=COLOR_TEXT,
        border_color_primary=COLOR_BORDER,
        border_color_primary_dark=COLOR_BORDER,
        background_fill_primary=COLOR_SURFACE_ALT,
        background_fill_primary_dark=COLOR_SURFACE_ALT,
        background_fill_secondary=COLOR_SURFACE,
        background_fill_secondary_dark=COLOR_SURFACE,
        color_accent=COLOR_PRIMARY,
        color_accent_soft="#818cf8",
        button_primary_background_fill=COLOR_PRIMARY,
        button_primary_background_fill_hover=COLOR_PRIMARY_DARK,
        button_primary_text_color="#ffffff",
        button_secondary_background_fill=COLOR_SURFACE_ALT,
        button_secondary_text_color=COLOR_TEXT,
        input_background_fill=COLOR_SURFACE,
        input_background_fill_dark=COLOR_SURFACE,
        input_border_color=COLOR_BORDER,
        block_title_text_color=COLOR_TEXT,
        block_label_text_color=COLOR_TEXT_MUTED,
        panel_background_fill=COLOR_SURFACE_ALT,
        panel_border_color=COLOR_BORDER,
        table_even_background_fill=COLOR_SURFACE,
        table_odd_background_fill=COLOR_SURFACE_ALT,
        checkbox_background_color=COLOR_SURFACE,
        checkbox_border_color=COLOR_BORDER,
        slider_color=COLOR_PRIMARY,
        stat_background_fill=COLOR_SURFACE_ALT,
    )


# ── Custom CSS ────────────────────────────────────────────────────────────────

CUSTOM_CSS = f"""
/* ── Global resets ──────────────────────────────────────────── */
.gradio-container {{
    max-width: 1400px !important;
    margin: 0 auto;
}}

/* ── Status pill badges ─────────────────────────────────────── */
.pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}}
.pill-green  {{ background: rgba(34,197,94,0.15);  color: {COLOR_SUCCESS}; }}
.pill-yellow {{ background: rgba(245,158,11,0.15); color: {COLOR_WARNING}; }}
.pill-red    {{ background: rgba(239,68,68,0.15);  color: {COLOR_ERROR};   }}
.pill-blue   {{ background: rgba(99,102,241,0.15); color: {COLOR_PRIMARY}; }}
.pill-gray   {{ background: rgba(148,163,184,0.1); color: {COLOR_TEXT_MUTED}; }}

/* ── CLI mirror code block ──────────────────────────────────── */
.cli-mirror {{
    background: #0f0f1a;
    border: 1px solid {COLOR_BORDER};
    border-radius: {RADIUS};
    padding: {SPACING_MD};
    font-family: {FONT_MONO};
    font-size: 0.8rem;
    color: #a5f3fc;
    line-height: 1.6;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
}}

/* ── Section headers ────────────────────────────────────────── */
.section-header {{
    font-size: 1rem;
    font-weight: 700;
    color: {COLOR_TEXT};
    border-bottom: 1px solid {COLOR_BORDER};
    padding-bottom: {SPACING_SM};
    margin-bottom: {SPACING_MD};
}}

/* ── Metric cards ───────────────────────────────────────────── */
.metric-card {{
    background: {COLOR_SURFACE_ALT};
    border: 1px solid {COLOR_BORDER};
    border-radius: {RADIUS};
    padding: {SPACING_MD};
    text-align: center;
}}
.metric-value {{
    font-size: 1.75rem;
    font-weight: 800;
    color: {COLOR_PRIMARY};
}}
.metric-label {{
    font-size: 0.75rem;
    color: {COLOR_TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

/* ── Warning / error banners ────────────────────────────────── */
.banner-warning {{
    background: rgba(245,158,11,0.1);
    border-left: 3px solid {COLOR_WARNING};
    padding: {SPACING_SM} {SPACING_MD};
    border-radius: 0 {RADIUS} {RADIUS} 0;
    color: {COLOR_TEXT};
    font-size: 0.875rem;
}}
.banner-error {{
    background: rgba(239,68,68,0.1);
    border-left: 3px solid {COLOR_ERROR};
    padding: {SPACING_SM} {SPACING_MD};
    border-radius: 0 {RADIUS} {RADIUS} 0;
    color: {COLOR_TEXT};
    font-size: 0.875rem;
}}
"""
