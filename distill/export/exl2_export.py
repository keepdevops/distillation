"""EXL2 quantization export.

EXL2 (ExLlamaV2 format) provides the best quality/speed ratio for single-GPU
inference, supporting mixed-precision per-layer quantization (2–8 bit).
Requires exllamav2 to be installed and a CUDA GPU.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _has_exllamav2() -> bool:
    try:
        import exllamav2  # type: ignore[import]
        return True
    except ImportError:
        return False


def _require_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def export_exl2(
    model_path: str,
    output_dir: str,
    bits: float = 4.0,
    head_bits: int = 6,
    measurement_path: str | None = None,
) -> dict[str, Any]:
    """Export a model to EXL2 format.

    Args:
        model_path:        HF model path (merged, not LoRA adapter).
        output_dir:        Where to write EXL2 quantized weights.
        bits:              Average bits per weight (e.g. 4.0, 4.65, 6.0).
        head_bits:         Bits for the LM head (usually 6 or 8).
        measurement_path:  Optional path to cached layer measurements.

    Requires: pip install exllamav2
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not _require_cuda():
        msg = "EXL2 quantization requires a CUDA GPU."
        logger.warning(msg)
        return {"output_path": "", "format": "exl2", "error": msg}

    if not _has_exllamav2():
        msg = "exllamav2 not installed. Run: pip install exllamav2"
        logger.error(msg)
        return {"output_path": "", "format": "exl2", "error": msg}

    try:
        # EXL2 quantization is best driven via the convert.py script
        # bundled with exllamav2
        import exllamav2
        exl2_dir = Path(exllamav2.__file__).parent
        convert_script = exl2_dir / "convert.py"

        if not convert_script.exists():
            raise FileNotFoundError(f"exllamav2 convert.py not found at {convert_script}")

        meas_dir = out / "measurement"
        meas_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, str(convert_script),
            "-i", model_path,
            "-o", str(out),
            "-cf", str(meas_dir),
            "-b", str(bits),
            "-hb", str(head_bits),
        ]
        if measurement_path:
            cmd += ["-m", measurement_path]

        logger.info("Running EXL2 conversion: bits=%.2f, head=%d", bits, head_bits)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])

        logger.info("EXL2 export complete: %s", out)
        return {"output_path": str(out), "format": "exl2", "error": ""}

    except Exception as exc:
        logger.error("EXL2 export failed: %s", exc)
        return {"output_path": "", "format": "exl2", "error": str(exc)}


def exl2_bits_presets() -> dict[str, float]:
    """Named quality presets for EXL2 bits-per-weight."""
    return {
        "2.2 bpw (smallest)": 2.2,
        "3.0 bpw":            3.0,
        "4.0 bpw (balanced)": 4.0,
        "4.65 bpw":           4.65,
        "6.0 bpw (quality)":  6.0,
        "8.0 bpw (near-fp16)": 8.0,
    }
