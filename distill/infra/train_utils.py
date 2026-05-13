"""
Shared training utilities for distillation scripts.

Centralises device detection, metric logging, pause-flag handling, and
model loading so that all backends behave consistently.
"""
from __future__ import annotations

import json
from pathlib import Path


def get_device():
    """Return the best available torch device: mps > cuda > cpu."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_device_cache(device) -> None:
    """Free any cached memory for the given device (MPS or CUDA)."""
    import torch
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def check_pause_flag(output_dir: Path) -> bool:
    """Return True if pause.flag exists in output_dir (PauseFlagCallback protocol)."""
    flag_path = Path(output_dir) / "pause.flag"
    if not flag_path.exists():
        return False
    try:
        with open(flag_path) as f:
            info = json.load(f)
        reason = info.get("reason", "unknown")
    except (json.JSONDecodeError, OSError):
        reason = "pause.flag"
    import logging
    logging.getLogger(__name__).info("pause.flag detected (reason=%s). Stopping.", reason)
    return True


def write_metric(metrics_path: Path, step: int, epoch: float, **kwargs) -> None:
    """Append a metrics row to metrics.jsonl (MetricsCallback format)."""
    row = {"step": step, "epoch": epoch, **kwargs}
    with open(metrics_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_student_model(checkpoint_dir: Path, student_id: str, cache_dir, offline: bool,
                       device):
    """Load model + tokenizer from checkpoint_dir, handling LoRA adapter checkpoints.

    Returns:
        (model, tokenizer) — model is merged, eval mode, on device.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_dir = Path(checkpoint_dir)
    tok_dir = str(checkpoint_dir) if (checkpoint_dir / "tokenizer_config.json").exists() else student_id
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, cache_dir=cache_dir, local_files_only=offline)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only: left-pad so generated tokens are right-aligned

    is_adapter = (checkpoint_dir / "adapter_config.json").exists()
    if is_adapter:
        with open(checkpoint_dir / "adapter_config.json") as f:
            adapter_cfg = json.load(f)
        base_id = adapter_cfg.get("base_model_name_or_path", student_id)
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_id, dtype=torch.bfloat16, device_map="auto",
            cache_dir=cache_dir, local_files_only=offline,
        )
        model = PeftModel.from_pretrained(base, str(checkpoint_dir))
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir), dtype=torch.bfloat16, device_map="auto",
            cache_dir=cache_dir, local_files_only=offline,
        )
    model.to(device)
    model.eval()
    return model, tokenizer
