"""'Powered by distill_cpp' telemetry display component.

Shows whether the C++ extension is loaded, backend acceleration status,
and live struct-backed metrics summary.
"""
from __future__ import annotations

import gradio as gr


def _cpp_status_html() -> str:
    from distill.cpp import is_available
    cpp_ok = is_available()

    if cpp_ok:
        badge = '<span class="pill pill-green">⚡ distill_cpp loaded</span>'
        detail = "C++ pybind11 structs active — zero-copy metrics streaming enabled."
    else:
        badge = '<span class="pill pill-gray">○ distill_cpp not compiled</span>'
        detail = (
            "Pure-Python fallback active. "
            "Run <code>bash scripts/build_cpp.sh</code> to enable native acceleration."
        )

    # Hybrid backend status
    try:
        from distill.training.backends.hybrid_selector import select_backend, all_available_backends
        rec = select_backend()
        backends = all_available_backends()
        backend_html = (
            f'<div style="margin-top:.5rem;font-size:.8rem;color:#94a3b8">'
            f'  Available backends: <b style="color:#e2e8f0">{", ".join(backends)}</b><br>'
            f'  Recommended: <b style="color:#6366f1">{rec["backend"].upper()}</b>'
            f'  — {rec["rationale"]}'
            f'</div>'
        )
    except Exception:
        backend_html = ""

    # Thermal struct status
    try:
        from distill.backends.cpp_thermal_bridge import read_thermal_dict, build_hardware_profile_dict
        td = read_thermal_dict()
        hw = build_hardware_profile_dict()
        cpp_thermal = "C++" if hw.get("cpp_backed") else "Python"
        thermal_html = (
            f'<div style="margin-top:.4rem;font-size:.75rem;color:#94a3b8">'
            f'  Thermal path: <b>{cpp_thermal}</b> · '
            f'  Hardware: <b>{hw.get("device","?")} · {hw.get("ram_gb",0):.0f}GB</b>'
            f'</div>'
        )
    except Exception:
        thermal_html = ""

    return (
        f'<div style="background:#1e1e2e;border:1px solid #1e293b;border-radius:.5rem;'
        f'padding:.75rem 1rem;margin-bottom:.75rem">'
        f'  <div style="display:flex;gap:.75rem;align-items:center;flex-wrap:wrap">'
        f'    {badge}'
        f'    <span style="font-size:.8rem;color:#94a3b8">{detail}</span>'
        f'  </div>'
        f'  {backend_html}'
        f'  {thermal_html}'
        f'</div>'
    )


def render_cpp_stats() -> None:
    """Render the C++ status panel inside the current gr.Blocks context."""
    gr.HTML(value=_cpp_status_html, every=60)
