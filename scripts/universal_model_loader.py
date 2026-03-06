#!/usr/bin/env python3
"""
Universal model loader for Gradio UIs.
Supports PyTorch, MLX, GGUF (llama.cpp), and vLLM backends.
Auto-detects format and loads appropriately.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

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
        list(path.glob("*.safetensors")) or
        list(path.glob("model*.bin")) or
        (path / "pytorch_model.bin").exists() or
        (path / "adapter_model.bin").exists()  # LoRA adapters
    )

    if has_config and has_weights:
        # Could be either PyTorch or vLLM (vLLM uses HF format)
        # Default to PyTorch, vLLM can be explicitly selected
        return ModelFormat.PYTORCH

    return ModelFormat.UNKNOWN


class UniversalModelLoader:
    """Universal model loader supporting multiple backends."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.backend = None
        self.model_path = None

    def load(self, model_path: str, backend: Optional[str] = None) -> Tuple[bool, str]:
        """
        Load a model from the given path.

        Args:
            model_path: Path to model directory or file
            backend: Optional backend override ('pytorch', 'mlx', 'gguf', 'vllm')
                    If None, auto-detects

        Returns:
            (success: bool, message: str)
        """
        model_path = str(Path(model_path).resolve())

        # Detect format if not specified
        if backend is None:
            detected = detect_model_format(model_path)
            backend = detected.value
            LOG.info(f"Auto-detected format: {backend}")

        try:
            if backend == ModelFormat.PYTORCH:
                return self._load_pytorch(model_path)
            elif backend == ModelFormat.MLX:
                return self._load_mlx(model_path)
            elif backend == ModelFormat.GGUF:
                return self._load_gguf(model_path)
            elif backend == ModelFormat.VLLM:
                return self._load_vllm(model_path)
            else:
                return False, f"Unknown or unsupported format: {backend}"
        except Exception as e:
            LOG.error(f"Failed to load model: {e}", exc_info=True)
            return False, f"Load error: {str(e)}"

    def _load_pytorch(self, model_path: str) -> Tuple[bool, str]:
        """Load PyTorch/HuggingFace model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Check for LoRA adapters
            adapter_config = Path(model_path) / "adapter_config.json"
            if adapter_config.exists():
                # Load base model and apply adapter
                import json
                with open(adapter_config) as f:
                    adapter_info = json.load(f)
                base_model = adapter_info.get("base_model_name_or_path")

                if not base_model:
                    return False, "LoRA adapter found but no base_model_name_or_path"

                LOG.info(f"Loading base model {base_model} with LoRA adapter")
                from peft import PeftModel

                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                base = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if not torch.backends.mps.is_available() else None
                )
                model = PeftModel.from_pretrained(base, model_path)
                model = model.merge_and_unload()  # Merge for inference
            else:
                # Regular model
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if not torch.backends.mps.is_available() else None,
                    local_files_only=True
                )

            # Move to device if MPS
            if torch.backends.mps.is_available():
                model = model.to("mps")

            self.model = model
            self.tokenizer = tokenizer
            self.backend = ModelFormat.PYTORCH
            self.model_path = model_path

            return True, f"Loaded PyTorch model: {Path(model_path).name}"

        except ImportError as e:
            return False, f"Missing dependencies for PyTorch: {e}"
        except Exception as e:
            return False, f"PyTorch load error: {e}"

    def _load_mlx(self, model_path: str) -> Tuple[bool, str]:
        """Load MLX model."""
        try:
            import mlx.core as mx
            import mlx_lm

            path = Path(model_path)

            # Check for mlx_lm compatible format (config.json + weights)
            if (path / "config.json").exists():
                # Use mlx_lm.load() for full models
                model, tokenizer = mlx_lm.load(str(path))
                self.model = model
                self.tokenizer = tokenizer
                self.backend = ModelFormat.MLX
                self.model_path = model_path
                return True, f"Loaded MLX model: {path.name}"

            # Raw weights file - need to load manually
            npz_files = list(path.glob("*.npz"))
            if npz_files:
                # For raw weights, we need the base model config
                # This is a limitation - raw .npz files need more context
                return False, "Raw .npz weights found but no config.json. Need full MLX model with config."

            return False, "No MLX model files found"

        except ImportError:
            return False, "MLX not installed. Install with: pip install mlx mlx-lm"
        except Exception as e:
            return False, f"MLX load error: {e}"

    def _load_gguf(self, model_path: str) -> Tuple[bool, str]:
        """Load GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama

            path = Path(model_path)

            # If directory, find first .gguf file
            if path.is_dir():
                gguf_files = list(path.glob("*.gguf"))
                if not gguf_files:
                    return False, "No .gguf files found in directory"
                gguf_path = gguf_files[0]
            else:
                gguf_path = path

            # Load with llama.cpp
            # n_ctx: context window, n_gpu_layers: for GPU acceleration
            model = Llama(
                model_path=str(gguf_path),
                n_ctx=2048,  # Context window
                n_gpu_layers=-1,  # Use all GPU layers if available (Metal on Mac)
                verbose=False
            )

            self.model = model
            self.tokenizer = None  # GGUF handles tokenization internally
            self.backend = ModelFormat.GGUF
            self.model_path = str(gguf_path)

            return True, f"Loaded GGUF model: {gguf_path.name}"

        except ImportError:
            return False, "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
        except Exception as e:
            return False, f"GGUF load error: {e}"

    def _load_vllm(self, model_path: str) -> Tuple[bool, str]:
        """Load model with vLLM for faster inference."""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            # vLLM uses HuggingFace format
            model = LLM(
                model=model_path,
                dtype="bfloat16",
                max_model_len=2048,
                gpu_memory_utilization=0.9
            )

            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

            self.model = model
            self.tokenizer = tokenizer
            self.backend = ModelFormat.VLLM
            self.model_path = model_path

            return True, f"Loaded vLLM model: {Path(model_path).name}"

        except ImportError:
            return False, "vLLM not installed. Install with: pip install vllm"
        except Exception as e:
            return False, f"vLLM load error: {e}"

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text (without prompt)
        """
        if self.model is None:
            return "Error: No model loaded"

        try:
            if self.backend == ModelFormat.PYTORCH:
                return self._generate_pytorch(prompt, max_tokens, temperature)
            elif self.backend == ModelFormat.MLX:
                return self._generate_mlx(prompt, max_tokens, temperature)
            elif self.backend == ModelFormat.GGUF:
                return self._generate_gguf(prompt, max_tokens, temperature)
            elif self.backend == ModelFormat.VLLM:
                return self._generate_vllm(prompt, max_tokens, temperature)
            else:
                return f"Error: Unsupported backend {self.backend}"
        except Exception as e:
            LOG.error(f"Generation error: {e}", exc_info=True)
            return f"Generation error: {str(e)}"

    def _generate_pytorch(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with PyTorch model."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.model.device.type in ("mps", "cuda"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_mlx(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with MLX model."""
        import mlx_lm

        response = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False
        )

        # mlx_lm.generate returns the full text including prompt
        # Remove the prompt to return only generated text
        if response.startswith(prompt):
            return response[len(prompt):]
        return response

    def _generate_gguf(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with GGUF model via llama.cpp."""
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            echo=False  # Don't echo the prompt
        )

        return output["choices"][0]["text"]

    def _generate_vllm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with vLLM."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens
        )

        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def get_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        if self.model is None:
            return {"loaded": False}

        info = {
            "loaded": True,
            "backend": self.backend,
            "path": self.model_path,
            "name": Path(self.model_path).name if self.model_path else "Unknown"
        }

        # Add backend-specific info
        if self.backend == ModelFormat.PYTORCH:
            import torch
            info["device"] = str(self.model.device)
            info["dtype"] = str(self.model.dtype)

        return info
