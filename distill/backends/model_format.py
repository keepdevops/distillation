#!/usr/bin/env python3
"""
Model format detection for universal loader.
"""

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


class ModelFormat:
    """Enum for model formats."""

    PYTORCH = "pytorch"
    MLX = "mlx"
    GGUF = "gguf"
    VLLM = "vllm"
    UNKNOWN = "unknown"


def detect_model_format(model_path: str) -> ModelFormat:
    """
    Detect the model format based on files in the directory.

    Returns:
        ModelFormat enum value
    """
    path = Path(model_path)

    # Single GGUF file
    if path.is_file() and path.suffix == ".gguf":
        return ModelFormat.GGUF

    if not path.is_dir():
        LOG.warning(f"Path does not exist or is not a directory: {model_path}")
        return ModelFormat.UNKNOWN

    # Check for GGUF files in directory
    if list(path.glob("*.gguf")):
        return ModelFormat.GGUF

    # Check for MLX weights
    if (path / "mlx_student_weights.npz").exists() or list(path.glob("*.npz")):
        # If there's also a config.json, might be MLX quantized format
        if (path / "config.json").exists():
            return ModelFormat.MLX
        # Raw MLX weights
        return ModelFormat.MLX

    # Check for PyTorch/HuggingFace format
    has_config = (path / "config.json").exists()
    has_weights = (
        list(path.glob("*.safetensors"))
        or list(path.glob("model*.bin"))
        or (path / "pytorch_model.bin").exists()
        or (path / "adapter_model.bin").exists()  # LoRA adapters
    )

    if has_config and has_weights:
        # Could be either PyTorch or vLLM (vLLM uses HF format)
        # Default to PyTorch, vLLM can be explicitly selected
        return ModelFormat.PYTORCH

    return ModelFormat.UNKNOWN
