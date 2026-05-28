"""ONNX export for cross-platform and edge deployment.

Uses optimum[exporters] for LLM-optimized ONNX graphs with optional
INT8/INT4 quantization for size reduction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def export_onnx(
    model_path: str,
    output_dir: str,
    task: str = "text-generation-with-past",
    dtype: str = "fp32",
    device: str = "cpu",
    opset: int = 17,
) -> dict[str, Any]:
    """Export a causal LM to ONNX using Optimum.

    Args:
        model_path:  HF model path or ID.
        output_dir:  Where to write ONNX model files.
        task:        ONNX export task (text-generation-with-past is standard).
        dtype:       "fp32", "fp16", "int8", or "int4".
        device:      "cpu" or "cuda".
        opset:       ONNX opset version.

    Requires: pip install optimum[exporters]
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from optimum.exporters.onnx import main_export  # type: ignore[import]
    except ImportError:
        msg = "optimum not installed. Run: pip install optimum[exporters]"
        logger.error(msg)
        return {"output_path": "", "format": "onnx", "error": msg}

    try:
        logger.info("Exporting %s → ONNX (task=%s, dtype=%s)", model_path, task, dtype)
        main_export(
            model_name_or_path=model_path,
            output=out,
            task=task,
            opset=opset,
            device=device,
            fp16=(dtype == "fp16"),
            int8=(dtype == "int8"),
            optimize=None,
        )
        logger.info("ONNX export complete: %s", out)

        # Apply post-export quantization if int4 requested
        if dtype == "int4":
            _apply_onnx_int4(out)

        onnx_files = list(out.glob("*.onnx"))
        return {
            "output_path": str(out),
            "format":      "onnx",
            "files":       [str(f) for f in onnx_files],
            "error":       "",
        }
    except Exception as exc:
        logger.error("ONNX export failed: %s", exc)
        return {"output_path": "", "format": "onnx", "error": str(exc)}


def _apply_onnx_int4(output_dir: Path) -> None:
    """Apply ONNX INT4 weight-only quantization via onnxruntime."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore[import]
        for onnx_file in output_dir.glob("*.onnx"):
            q_file = output_dir / (onnx_file.stem + "_int4.onnx")
            logger.info("Applying INT4 quantization: %s", onnx_file.name)
            quantize_dynamic(str(onnx_file), str(q_file), weight_type=QuantType.QInt8)
    except ImportError:
        logger.warning("onnxruntime not available for INT4 quantization")
    except Exception as exc:
        logger.warning("INT4 quantization failed: %s", exc)


def estimate_onnx_size(model_path: str) -> str:
    """Rough estimate of ONNX export size from HF model config."""
    try:
        import json
        cfg_path = Path(model_path) / "config.json"
        if not cfg_path.exists():
            return "unknown"
        cfg = json.loads(cfg_path.read_text())
        hidden = cfg.get("hidden_size", 0)
        layers = cfg.get("num_hidden_layers", 0)
        vocab  = cfg.get("vocab_size", 0)
        if hidden and layers and vocab:
            # Very rough: 4 bytes * ~12 * hidden^2 * layers
            params = 12 * hidden * hidden * layers + vocab * hidden
            size_gb = params * 4 / (1024 ** 3)
            return f"~{size_gb:.1f} GB (FP32)"
    except Exception:
        pass
    return "unknown"
