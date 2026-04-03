"""Perplexity eval utilities: step detection, loss computation, model loading."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def detect_step(checkpoint_dir) -> Optional[int]:
    """Infer step number from checkpoint dir name (e.g. checkpoint-80 → 80)."""
    name = Path(checkpoint_dir).name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[-1])
        except ValueError:
            pass
    return None


def last_step_in_jsonl(jsonl_path) -> int:
    if not Path(jsonl_path).exists():
        return 0
    last = 0
    try:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    last = max(last, row.get("step", 0))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return last


@torch.no_grad()
def eval_loss(model, tokenizer, texts, max_length, batch_size, device) -> Optional[float]:
    """Compute mean token-level cross-entropy loss over texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return None
    return total_loss / total_tokens


@torch.no_grad()
def eval_model_at_path(model_id_or_path, tokenizer, texts, max_length, batch_size, device,
                       cache_dir=None, offline=False) -> Optional[float]:
    """Load a model from a path or HF id, eval loss, then unload."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
        local_files_only=offline,
    )
    m.to(device)
    loss = eval_loss(m, tokenizer, texts, max_length, batch_size, device)
    del m
    if device.type == "mps":
        torch.mps.empty_cache()
    return loss
