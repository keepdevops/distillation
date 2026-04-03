#!/usr/bin/env python3
"""
MLX inference/eval utilities for Apple Silicon (M-series).

Provides batched perplexity computation and text generation using mlx-lm,
which uses native Metal kernels and unified memory — significantly faster
than PyTorch MPS for forward-only eval on M3 Max.

Designed to be imported by run_eval.py, run_benchmarks.py, and eval_quality.py.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


def is_mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        from mlx_lm import load  # noqa: F401
        return True
    except ImportError:
        return False


def load_mlx_model(model_path: str, student_name: str | None = None):
    """Load model + tokenizer via mlx-lm. Returns (model, tokenizer).

    Handles three cases:
    1. Normal HF-format dir (has config.json) — load directly.
    2. MLX training output dir with mlx_q* quantized subdir — load quant.
    3. MLX training output dir with mlx_student_weights.npz — load base
       student from HF (name from distill_config.json or student_name arg)
       then apply the trained NPZ weights.
    """
    import json as _json
    from pathlib import Path as _Path
    from mlx_lm import load

    p = _Path(model_path)

    if not (p / "config.json").exists():
        # Case 2: quantized subdir present
        quant_dirs = sorted(p.glob("mlx_q*/"))
        if quant_dirs:
            p = quant_dirs[-1]
            logger.info("Using MLX quantized subdir: %s", p)
        # Case 3: raw NPZ weights from distill_mlx.py (LoRA-trained)
        elif (p / "mlx_student_weights.npz").exists():
            cfg_file = p / "distill_config.json"
            base_name = None
            lora_r = 8
            if cfg_file.exists():
                cfg = _json.loads(cfg_file.read_text())
                base_name = cfg.get("student")
                lora_r = cfg.get("lora_r", 8)
            base_name = base_name or student_name
            if not base_name:
                raise ValueError(
                    f"MLX training output at {p} has no config.json; pass "
                    "--student <model-id> so the base architecture can be loaded"
                )
            logger.info(
                "Loading MLX base model %s with LoRA r=%d and applying trained weights",
                base_name, lora_r,
            )
            from mlx_lm.tuner import linear_to_lora_layers
            model, tokenizer = load(base_name)
            num_blocks = len(list(model.model.layers)) if hasattr(model, "model") else 24
            lora_config = {"rank": lora_r, "scale": 20.0, "dropout": 0.0}
            linear_to_lora_layers(model, num_blocks, lora_config)
            model.load_weights(str(p / "mlx_student_weights.npz"))
            model.eval()
            return model, tokenizer

    logger.info("Loading MLX model from %s", p)
    model, tokenizer = load(str(p))
    return model, tokenizer


def compute_mlx_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
    batch_size: int = 8,
) -> Optional[float]:
    """
    Batched perplexity (mean cross-entropy) over texts using MLX.

    Uses a single forward pass per batch — no autoregressive decode.
    On M3 Max this is 3-5× faster than PyTorch MPS for the same sequences.
    """
    import mlx.core as mx
    import mlx.nn as nn

    # Access underlying HF tokenizer for batch tokenization with padding
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    pad_id = getattr(hf_tok, "pad_token_id", None) or getattr(hf_tok, "eos_token_id", 0) or 0

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        enc = hf_tok(
            batch,
            return_tensors=None,
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = mx.array(enc["input_ids"])           # (B, T)
        attention_mask = mx.array(enc["attention_mask"]) # (B, T)

        logits = model(input_ids)  # (B, T, V)

        # Causal shift: position i predicts position i+1
        shift_logits = logits[:, :-1, :]                          # (B, T-1, V)
        shift_labels = input_ids[:, 1:]                           # (B, T-1)
        shift_mask   = attention_mask[:, 1:].astype(mx.float32)   # (B, T-1)

        # Per-token cross-entropy, then mask out padding
        token_loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
        ).reshape(shift_labels.shape)  # (B, T-1)

        masked_loss = token_loss * shift_mask
        mx.eval(masked_loss)

        total_loss += float(masked_loss.sum())
        total_tokens += int(shift_mask.sum())

        if (i // batch_size + 1) % 10 == 0:
            logger.info("  MLX perplexity: %d/%d batches", i // batch_size + 1, math.ceil(len(texts) / batch_size))

    if total_tokens == 0:
        return None
    return total_loss / total_tokens


def mlx_generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> list[str]:
    """
    Generate responses for a list of prompts using mlx-lm.

    mlx-lm does not support true batched autoregressive decode, but individual
    generation is fast enough on M3 Max (~150 tok/s for 0.5B, ~80 tok/s for 1.5B)
    that sequential generation over dozens of prompts is practical.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)
    responses = []
    for idx, prompt in enumerate(prompts):
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        # mlx-lm ≥0.10 returns only the generated text; older versions return
        # the full string including the prompt — strip prompt prefix if present.
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        responses.append(text)

        if (idx + 1) % 10 == 0:
            logger.info("  MLX generation: %d/%d prompts done", idx + 1, len(prompts))

    return responses
