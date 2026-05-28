"""Merge a LoRA adapter into the base model before export.

Required for GGUF, ONNX, and any format that needs a single merged checkpoint.
Supports PEFT LoRA adapters saved via save_pretrained().
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def merge_lora_into_base(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    dtype: str = "float16",
) -> Path:
    """Merge a LoRA adapter into the base model and save merged weights.

    Args:
        base_model_id: HF model ID or local path for the base model.
        adapter_path:  Path to the PEFT adapter directory.
        output_dir:    Where to write the merged model.
        dtype:         Output dtype ("float16", "bfloat16", "float32").

    Returns:
        Path to the merged model directory.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    logger.info("Loading base model: %s", base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    logger.info("Loading LoRA adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()

    logger.info("Saving merged model to %s", out)
    model.save_pretrained(out, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(out)

    del model
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    logger.info("Merge complete: %s", out)
    return out


def is_lora_adapter(path: str) -> bool:
    """Return True if path looks like a PEFT LoRA adapter directory."""
    p = Path(path)
    return (p / "adapter_config.json").exists() or (p / "adapter_model.safetensors").exists()


def find_base_model_from_adapter(adapter_path: str) -> str | None:
    """Read base_model_name_or_path from adapter_config.json."""
    import json
    cfg = Path(adapter_path) / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text())
        return data.get("base_model_name_or_path")
    except Exception as exc:
        logger.warning("Could not read adapter_config.json: %s", exc)
        return None
