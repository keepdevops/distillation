"""Standalone utilities for MiniLLM distillation."""
from __future__ import annotations

# Threshold for "likely clipped" completions.
# Calibrated for 256-token max: 256 × ~3.3 chars/tok × 0.97 ≈ 819 chars; 800 gives a
# small buffer to reliably catch outputs clipped at the hard limit.
# Update this constant if you change --max_new_tokens significantly.
_MAX_NATURAL_CHARS = 800


def response_quality_reward(completions: list, **kwargs) -> list:
    """
    Discriminative reward for instruction-following quality.

    Provides variance within GRPO completion groups so the advantage signal is
    non-zero even without an external verifier:
      -1.0  mode collapse (empty / near-empty response)
      -0.5  over-generation (completion likely clipped at max_completion_length)
      +0.5  natural termination with reasonable content

    _MAX_NATURAL_CHARS should be kept slightly below max_new_tokens * ~3.5 chars/tok
    to catch clipped outputs before the hard limit.

    TRL passes completions as plain strings for non-chat datasets, or as
    list[dict] (conversational format) for instruct/chat models.
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
        else:
            text = completion
        text = text.strip()
        n = len(text)
        if n < 10:
            rewards.append(-1.0)
        elif n > _MAX_NATURAL_CHARS:
            rewards.append(-0.5)
        else:
            rewards.append(0.5)
    return rewards


def detect_attn_impl() -> str:
    """Return the best available attention implementation for the current environment."""
    import torch
    try:
        import flash_attn  # noqa: F401
        print("✓ Flash Attention 2 detected, enabling (2-3x speedup)")
        return "flash_attention_2"
    except ImportError:
        print("Flash Attention 2 not available (CUDA only). Install for 2-3x speedup:")
        print("  pip install flash-attn --no-build-isolation")

    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        print("✓ Using SDPA attention (fused kernel, faster than eager on MPS)")
        return "sdpa"

    return "eager"
