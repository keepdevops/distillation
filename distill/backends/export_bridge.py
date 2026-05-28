"""Python adapter for ExportFormatSpec and ExportManifest C++ structs.

Bridges between UI format selections (string labels or keys) and the
distill_cpp ExportFormatSpec / ExportManifest structs.
Falls back to plain dicts when the C++ module is not compiled.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _try_cpp_export():
    try:
        import distill_cpp  # type: ignore[import]
        return (
            distill_cpp.ExportFormatSpec,
            distill_cpp.ExportResult,
            distill_cpp.ExportManifest,
        )
    except (ImportError, AttributeError):
        return None, None, None


def build_format_spec(
    format_key: str,
    quant_method: str = "q4_k_m",
    bits: int = 4,
    group_size: int = 128,
    merge_lora: bool = True,
    output_dir: str = "exports",
    push_to_hub: bool = False,
    hub_repo_id: str = "",
    optimize_for: str = "balanced",
) -> dict[str, Any]:
    """Return a validated ExportFormatSpec dict."""
    from distill.config.schemas import ExportConfig
    cfg = ExportConfig(
        format=format_key, quant_method=quant_method, bits=bits,
        group_size=group_size, merge_lora=merge_lora, output_dir=output_dir,
        push_to_hub=push_to_hub, hub_repo_id=hub_repo_id, optimize_for=optimize_for,
    )

    Spec, _, _ = _try_cpp_export()
    if Spec is not None:
        try:
            s = Spec()
            s.format_key   = cfg.format
            s.quant_method = cfg.quant_method
            s.bits         = cfg.bits
            s.group_size   = cfg.group_size
            s.merge_lora   = cfg.merge_lora
            s.push_to_hub  = cfg.push_to_hub
            s.output_dir   = cfg.output_dir
            s.hub_repo_id  = cfg.hub_repo_id
            s.optimize_for = cfg.optimize_for
            return {**s.to_dict(), "cpp_backed": True}
        except Exception as exc:
            logger.warning("C++ ExportFormatSpec failed: %s", exc)

    return {
        "format_key":   cfg.format,
        "quant_method": cfg.quant_method,
        "bits":         cfg.bits,
        "group_size":   cfg.group_size,
        "merge_lora":   cfg.merge_lora,
        "push_to_hub":  cfg.push_to_hub,
        "output_dir":   cfg.output_dir,
        "optimize_for": cfg.optimize_for,
        "cpp_backed":   False,
    }


def build_result(
    format_key: str,
    output_path: str,
    success: bool,
    error: str = "",
    elapsed_sec: float = 0.0,
) -> dict[str, Any]:
    """Wrap a raw export outcome in an ExportResult dict."""
    _, Result, _ = _try_cpp_export()

    size_mb = 0.0
    p = Path(output_path)
    if p.is_file():
        size_mb = p.stat().st_size / (1024 ** 2)
    elif p.is_dir():
        size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 2)

    if Result is not None:
        try:
            r = Result()
            r.format_key  = format_key
            r.output_path = output_path
            r.success     = success
            r.error       = error
            r.size_mb     = size_mb
            r.elapsed_sec = elapsed_sec
            return {**r.to_dict(), "cpp_backed": True,
                    "status_icon": r.status_icon()}
        except Exception as exc:
            logger.warning("C++ ExportResult failed: %s", exc)

    return {
        "format_key":  format_key,
        "output_path": output_path,
        "success":     success,
        "error":       error,
        "size_mb":     round(size_mb, 2),
        "elapsed_sec": round(elapsed_sec, 1),
        "status_icon": "✅" if success else "❌",
        "cpp_backed":  False,
    }


def build_manifest(
    model_id: str,
    source_path: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bundle export results into a manifest dict."""
    _, _, Manifest = _try_cpp_export()

    import datetime
    created = datetime.datetime.now().isoformat(timespec="seconds")

    if Manifest is not None:
        try:
            import distill_cpp
            m = Manifest()
            m.model_id    = model_id
            m.source_path = source_path
            m.created_at  = created
            for r_dict in results:
                r = distill_cpp.ExportResult()
                r.format_key  = r_dict.get("format_key", "")
                r.output_path = r_dict.get("output_path", "")
                r.success     = bool(r_dict.get("success", False))
                r.error       = r_dict.get("error", "")
                r.size_mb     = float(r_dict.get("size_mb", 0.0))
                m.add_result(r)
            return {
                "model_id":     m.model_id,
                "source_path":  m.source_path,
                "created_at":   m.created_at,
                "total_formats": len(m),
                "succeeded":    m.success_count(),
                "failed":       m.failure_count(),
                "total_size_mb": m.total_size_mb(),
                "cpp_backed":   True,
            }
        except Exception as exc:
            logger.warning("C++ ExportManifest failed: %s", exc)

    n_ok = sum(1 for r in results if r.get("success"))
    return {
        "model_id":      model_id,
        "source_path":   source_path,
        "created_at":    created,
        "total_formats": len(results),
        "succeeded":     n_ok,
        "failed":        len(results) - n_ok,
        "total_size_mb": round(sum(r.get("size_mb", 0) for r in results), 2),
        "cpp_backed":    False,
    }


def labels_to_specs(
    ui_labels: list[str],
    output_dir: str,
    quant_method: str = "q4_k_m",
    merge_lora: bool = True,
) -> list[dict[str, Any]]:
    """Convert a list of UI labels (from gr.CheckboxGroup) to ExportFormatSpec dicts."""
    from distill.ui.core.registry import registry
    specs = []
    for label in ui_labels:
        key = registry.export_label_to_key(label)
        if key:
            specs.append(build_format_spec(
                format_key=key,
                quant_method=quant_method,
                merge_lora=merge_lora,
                output_dir=str(Path(output_dir) / key),
            ))
    return specs
