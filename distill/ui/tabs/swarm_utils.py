"""Swarm export helper functions — artifact detection and config generation.

Kept separate from swarm_export.py to stay under 300 LOC and allow reuse
from CLI tools without pulling in Gradio.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXPORT_FORMATS = [
    "GGUF (llama.cpp)",
    "MLX",
    "CoreML",
    "Safetensors + HF Hub",
    "AWQ (4-bit GPU)",
    "GPTQ (4-bit GPU)",
    "EXL2 (high-perf GPU)",
    "vLLM config",
    "ONNX",
]

# Well-known output locations to scan for trained artifacts
_SCAN_ROOTS = [
    Path("/Users/Shared/llama/models"),
    Path.home() / "distilled",
    Path.home() / "distill" / "outputs",
    Path.cwd() / "outputs",
]


def detect_artifacts_summary() -> tuple[str, list[str]]:
    """Scan known paths for trained model artifacts.

    Returns (markdown_summary, list_of_paths).
    Reuses distill.infra.artifact_detector where available.
    """
    try:
        from distill.infra.artifact_detector import detect_artifacts
        artifacts = detect_artifacts()
        if not artifacts:
            return "_No artifacts found in standard output directories._", []
        lines = ["| Path | Format | Size |", "|---|---|---|"]
        paths = []
        for a in artifacts:
            path = str(a.get("path", ""))
            fmt  = a.get("format", "unknown")
            size = _dir_size_str(path)
            lines.append(f"| `{path}` | {fmt} | {size} |")
            paths.append(path)
        return "\n".join(lines), paths
    except ImportError:
        pass

    # Fallback: simple filesystem scan
    found: list[str] = []
    for root in _SCAN_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.gguf"):
            found.append(str(p))
        for p in root.glob("*/config.json"):
            found.append(str(p.parent))

    if not found:
        return "_No artifacts found. Run a training job first._", []

    lines = ["| Path | Size |", "|---|---|"]
    for f in found[:20]:
        lines.append(f"| `{f}` | {_dir_size_str(f)} |")
    return "\n".join(lines), found[:20]


def generate_swarm_config(
    model_path: str,
    formats: list[str],
    system_prompt: str,
    quant_method: str,
    output_dir: str,
) -> dict[str, Any]:
    """Generate a matrix swarm deployment config dict."""
    backends: list[dict] = []

    fmt_map = {
        "GGUF (llama.cpp)": {"backend": "llama_cpp", "format": "gguf"},
        "MLX":              {"backend": "mlx",        "format": "mlx_weights"},
        "CoreML":           {"backend": "coreml",     "format": "mlpackage"},
        "Safetensors + HF Hub": {"backend": "pytorch", "format": "safetensors"},
        "AWQ (4-bit GPU)":  {"backend": "awq",        "format": "awq_safetensors"},
        "GPTQ (4-bit GPU)": {"backend": "gptq",       "format": "gptq_safetensors"},
        "EXL2 (high-perf GPU)": {"backend": "exllama2", "format": "exl2"},
        "vLLM config":      {"backend": "vllm",       "format": "safetensors"},
        "ONNX":             {"backend": "onnxruntime", "format": "onnx"},
    }

    for fmt in formats:
        if fmt in fmt_map:
            entry = fmt_map[fmt].copy()
            entry["model_path"] = str(Path(output_dir) / Path(model_path).name)
            entry["quant_method"] = quant_method
            backends.append(entry)

    return {
        "swarm_version": "1.0",
        "model_id": Path(model_path).name,
        "source_path": model_path,
        "system_prompt": system_prompt,
        "quant_method": quant_method,
        "output_dir": output_dir,
        "backends": backends,
        "metadata": {
            "generated_by": "distill.ui.tabs.swarm_export",
            "format_count": len(backends),
        },
    }


def _dir_size_str(path: str) -> str:
    """Return human-readable size of a file or directory."""
    try:
        p = Path(path)
        if p.is_file():
            size = p.stat().st_size
        else:
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except Exception:
        return "?"
