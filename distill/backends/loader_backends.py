#!/usr/bin/env python3
"""
Backend-specific load and generate functions for the universal model loader.
Supports PyTorch, MLX, GGUF (llama.cpp), and vLLM backends.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

from transformers import GenerationConfig

from .model_format import ModelFormat

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load functions — each returns (success: bool, message: str, model, tokenizer)
# ---------------------------------------------------------------------------


def load_pytorch(model_path: str) -> Tuple[bool, str, Any, Any]:
    """Load a PyTorch/HuggingFace model, including LoRA adapters."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_config_path = Path(model_path) / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                adapter_info = json.load(f)
            base_model = adapter_info.get("base_model_name_or_path")

            if not base_model:
                return False, "LoRA adapter found but no base_model_name_or_path", None, None

            LOG.info(f"Loading base model {base_model} with LoRA adapter")
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto" if not torch.backends.mps.is_available() else None,
            )
            model = PeftModel.from_pretrained(base, model_path)
            model = model.merge_and_unload()
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if not torch.backends.mps.is_available() else None,
                local_files_only=True,
            )

        if torch.backends.mps.is_available():
            model = model.to("mps")

        return True, f"Loaded PyTorch model: {Path(model_path).name}", model, tokenizer

    except ImportError as e:
        LOG.error(f"Missing dependencies for PyTorch: {e}", exc_info=True)
        return False, f"Missing dependencies for PyTorch: {e}", None, None
    except Exception as e:
        LOG.error(f"PyTorch load error: {e}", exc_info=True)
        return False, f"PyTorch load error: {e}", None, None


def load_mlx(model_path: str) -> Tuple[bool, str, Any, Any]:
    """Load an MLX model using mlx_lm."""
    try:
        import mlx_lm  # noqa: F401 — validates install before use

        path = Path(model_path)

        if (path / "config.json").exists():
            model, tokenizer = mlx_lm.load(str(path))
            return True, f"Loaded MLX model: {path.name}", model, tokenizer

        npz_files = list(path.glob("*.npz"))
        if npz_files:
            msg = (
                "Raw .npz weights found but no config.json. "
                "Need full MLX model with config."
            )
            LOG.error(msg)
            return False, msg, None, None

        LOG.error("No MLX model files found at %s", model_path)
        return False, "No MLX model files found", None, None

    except ImportError:
        msg = "MLX not installed. Install with: pip install mlx mlx-lm"
        LOG.error(msg)
        return False, msg, None, None
    except Exception as e:
        LOG.error(f"MLX load error: {e}", exc_info=True)
        return False, f"MLX load error: {e}", None, None


def load_gguf(model_path: str) -> Tuple[bool, str, Any, Any]:
    """Load a GGUF model via llama-cpp-python."""
    try:
        from llama_cpp import Llama

        path = Path(model_path)

        if path.is_dir():
            gguf_files = list(path.glob("*.gguf"))
            if not gguf_files:
                LOG.error("No .gguf files found in directory: %s", model_path)
                return False, "No .gguf files found in directory", None, None
            gguf_path = gguf_files[0]
        else:
            gguf_path = path

        model = Llama(
            model_path=str(gguf_path),
            n_ctx=2048,
            n_gpu_layers=-1,  # Use all GPU layers if available (Metal on Mac)
            verbose=False,
        )

        # GGUF handles tokenization internally; tokenizer is None
        return True, f"Loaded GGUF model: {gguf_path.name}", model, None

    except ImportError:
        msg = "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
        LOG.error(msg)
        return False, msg, None, None
    except Exception as e:
        LOG.error(f"GGUF load error: {e}", exc_info=True)
        return False, f"GGUF load error: {e}", None, None


def load_vllm(model_path: str) -> Tuple[bool, str, Any, Any]:
    """Load a model with vLLM for faster inference."""
    try:
        from vllm import LLM  # noqa: F401
        from transformers import AutoTokenizer

        model = LLM(
            model=model_path,
            dtype="bfloat16",
            max_model_len=2048,
            gpu_memory_utilization=0.9,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        return True, f"Loaded vLLM model: {Path(model_path).name}", model, tokenizer

    except ImportError:
        msg = "vLLM not installed. Install with: pip install vllm"
        LOG.error(msg)
        return False, msg, None, None
    except Exception as e:
        LOG.error(f"vLLM load error: {e}", exc_info=True)
        return False, f"vLLM load error: {e}", None, None


# ---------------------------------------------------------------------------
# Generate functions — each returns the generated text as a string
# ---------------------------------------------------------------------------


def generate_pytorch(model: Any, tokenizer: Any, prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate text using a PyTorch model."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    if model.device.type in ("mps", "cuda"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            ),
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_mlx(model: Any, tokenizer: Any, prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate text using an MLX model."""
    import mlx_lm

    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False,
    )

    # mlx_lm.generate returns the full text including the prompt
    if response.startswith(prompt):
        return response[len(prompt):]
    return response


def generate_gguf(model: Any, prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate text using a GGUF model via llama.cpp."""
    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        echo=False,
    )
    return output["choices"][0]["text"]


def generate_vllm(model: Any, prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate text using a vLLM model."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    outputs = model.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text
