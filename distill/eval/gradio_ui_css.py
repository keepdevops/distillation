"""CSS constants for the Universal Gradio UI."""

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────── */
.gradio-container { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important; }

/* ── Header banner ──────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%);
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 1.4rem 2rem;
    margin-bottom: 0.5rem;
}
.app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c6af7, #4fc3f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.25rem 0;
}
.app-header p { color: #8b8fa8; font-size: 0.9rem; margin: 0; }

/* ── Tabs ────────────────────────────────────────────────── */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em;
}
.tab-nav button.selected {
    border-bottom: 3px solid #7c6af7 !important;
    color: #7c6af7 !important;
}

/* ── Cards / panels ─────────────────────────────────────── */
.card {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
}

/* ── Status box ─────────────────────────────────────────── */
.status-ok  textarea { border-left: 4px solid #4caf50 !important; color: #4caf50 !important; }
.status-err textarea { border-left: 4px solid #f44336 !important; color: #f44336 !important; }

/* ── Buttons ─────────────────────────────────────────────── */
.btn-primary { border-radius: 8px !important; font-weight: 600 !important; }
.generate-row { gap: 1rem; align-items: flex-end; }

/* ── Output box ─────────────────────────────────────────── */
.output-box textarea {
    background: #13151f !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 8px !important;
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    font-size: 0.88rem !important;
}

/* ── Algo iframe container ───────────────────────────────── */
.algo-frame { border-radius: 10px; overflow: hidden; border: 1px solid #2a2d3e; }
"""
