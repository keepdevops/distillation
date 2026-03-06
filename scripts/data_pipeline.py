"""
Shared data pipeline utilities for distillation scripts.

Centralises dataset loading, prompt formatting, and tokenization so that
distill_mlx.py, distill_sft.py, and distill_minillm.py all behave
consistently and fixes propagate to every backend at once.
"""

import os
from pathlib import Path


def load_dataset_split(dataset_path, max_samples=None, cache_dir=None, offline=False):
    """Load HF dataset train split. Handles local disk, HF hub, and offline/air-gap mode.

    Args:
        dataset_path: HF hub ID (e.g. 'tatsu-lab/alpaca') or path to saved dataset on disk.
        max_samples:  If set, truncate to this many samples.
        cache_dir:    HF cache directory override.
        offline:      If True, never hit the network (air-gap mode).

    Returns:
        HF Dataset (train split).
    """
    if Path(dataset_path).exists():
        from datasets import load_from_disk
        ds = load_from_disk(dataset_path)
    elif offline or os.environ.get("HF_DATASETS_OFFLINE") == "1":
        cache_candidates = [
            Path("datasets_cache") / dataset_path.replace("/", "___"),
            Path("scripts/datasets_cache") / dataset_path.replace("/", "___"),
        ]
        for c in cache_candidates:
            if c.exists():
                from datasets import load_from_disk
                ds = load_from_disk(str(c))
                break
        else:
            raise FileNotFoundError(
                f"Offline mode: '{dataset_path}' not found in cache. "
                "Run scripts/cache_datasets.py first."
            )
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset_path, split="train", cache_dir=cache_dir)
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        return ds

    # For DatasetDict (load_from_disk may return one), extract train split
    if hasattr(ds, "__getitem__") and "train" in ds:
        ds = ds["train"]

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    return ds


def format_prompt_only(example):
    """Format an Alpaca-style example as a prompt string (no response).

    Used by: distill_sft.py (teacher label generation), distill_minillm.py.
    """
    prompt = example.get("instruction", example.get("prompt", "")).strip()
    inp = example.get("input", "").strip()
    if inp:
        prompt += "\n\nInput: " + inp
    prompt += "\n\n### Response:"
    return prompt


def format_prompt_full(example):
    """Format an Alpaca-style example as prompt + response (full sequence).

    Used by: distill_mlx.py (forward KD over complete sequences).
    """
    prompt = example.get("instruction", example.get("prompt", "")).strip()
    inp = example.get("input", "").strip()
    if inp:
        prompt += "\n\nInput: " + inp
    output = example.get("output", example.get("response", "")).strip()
    return prompt + "\n\n### Response:\n" + output


def pretokenize(tokenizer, texts, max_length=512):
    """Tokenize a list of texts into padded numpy arrays for fast batch indexing.

    Accepts an mlx_lm TokenizerWrapper (unwraps ._tokenizer) or a plain HF tokenizer.

    Returns:
        input_ids:      np.ndarray of shape (N, max_length), dtype int32
        attention_mask: np.ndarray of shape (N, max_length), dtype int32
    """
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    enc = hf_tok(
        texts,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return enc["input_ids"], enc["attention_mask"]
