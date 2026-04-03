#!/usr/bin/env python3
"""
Universal model loader for Gradio UIs.
Supports PyTorch, MLX, GGUF (llama.cpp), and vLLM backends.
Auto-detects format and loads appropriately.

This module re-exports ModelFormat and detect_model_format so that
existing callers such as:
    from ..backends.universal_loader import UniversalModelLoader, ModelFormat, detect_model_format
continue to work without modification.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Public re-exports — keep these so callers don't break
from .model_format import ModelFormat, detect_model_format  # noqa: F401
from .loader_backends import (  # noqa: F401
    load_pytorch,
    load_mlx,
    load_gguf,
    load_vllm,
    generate_pytorch,
    generate_mlx,
    generate_gguf,
    generate_vllm,
)

LOG = logging.getLogger(__name__)


class UniversalModelLoader:
    """Thin wrapper that delegates to backend-specific free functions."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.backend: Optional[str] = None
        self.model_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, model_path: str, backend: Optional[str] = None) -> Tuple[bool, str]:
        """
        Load a model from the given path.

        Args:
            model_path: Path to model directory or file.
            backend: Optional backend override ('pytorch', 'mlx', 'gguf', 'vllm').
                     If None, auto-detects from path contents.

        Returns:
            (success: bool, message: str)
        """
        model_path = str(Path(model_path).resolve())

        if backend is None:
            detected = detect_model_format(model_path)
            backend = detected.value
            LOG.info(f"Auto-detected format: {backend}")

        try:
            if backend == ModelFormat.PYTORCH:
                ok, msg, model, tokenizer = load_pytorch(model_path)
            elif backend == ModelFormat.MLX:
                ok, msg, model, tokenizer = load_mlx(model_path)
            elif backend == ModelFormat.GGUF:
                ok, msg, model, tokenizer = load_gguf(model_path)
            elif backend == ModelFormat.VLLM:
                ok, msg, model, tokenizer = load_vllm(model_path)
            else:
                return False, f"Unknown or unsupported format: {backend}"
        except Exception as e:
            LOG.error(f"Failed to load model: {e}", exc_info=True)
            return False, f"Load error: {str(e)}"

        if ok:
            self.model = model
            self.tokenizer = tokenizer
            self.backend = backend
            self.model_path = model_path

        return ok, msg

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        Generate text from a prompt.

        Returns:
            Generated text (without the prompt).
        """
        if self.model is None:
            return "Error: No model loaded"

        try:
            if self.backend == ModelFormat.PYTORCH:
                return generate_pytorch(self.model, self.tokenizer, prompt, max_tokens, temperature)
            elif self.backend == ModelFormat.MLX:
                return generate_mlx(self.model, self.tokenizer, prompt, max_tokens, temperature)
            elif self.backend == ModelFormat.GGUF:
                return generate_gguf(self.model, prompt, max_tokens, temperature)
            elif self.backend == ModelFormat.VLLM:
                return generate_vllm(self.model, prompt, max_tokens, temperature)
            else:
                return f"Error: Unsupported backend {self.backend}"
        except Exception as e:
            LOG.error(f"Generation error: {e}", exc_info=True)
            return f"Generation error: {str(e)}"

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        """Return metadata about the currently loaded model."""
        if self.model is None:
            return {"loaded": False}

        info: Dict[str, Any] = {
            "loaded": True,
            "backend": self.backend,
            "path": self.model_path,
            "name": Path(self.model_path).name if self.model_path else "Unknown",
        }

        if self.backend == ModelFormat.PYTORCH:
            import torch  # noqa: F401 — only imported when backend is pytorch
            info["device"] = str(self.model.device)
            info["dtype"] = str(self.model.dtype)

        return info
