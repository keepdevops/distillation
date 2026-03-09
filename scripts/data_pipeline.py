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
        # Try standard HF cache first (works if dataset was previously downloaded)
        try:
            _prev = os.environ.get("HF_DATASETS_OFFLINE")
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            from datasets import load_dataset as _load_dataset
            ds = _load_dataset(dataset_path, split="train", cache_dir=cache_dir)
            if _prev is None:
                del os.environ["HF_DATASETS_OFFLINE"]
            else:
                os.environ["HF_DATASETS_OFFLINE"] = _prev
            if max_samples and len(ds) > max_samples:
                ds = ds.select(range(max_samples))
            return ds
        except Exception:
            pass
        # Fall back to explicit disk cache (created by cache_datasets.py / airgap.py prepare)
        cache_candidates = [
            Path("datasets_cache") / dataset_path.replace("/", "___"),
            Path("scripts/datasets_cache") / dataset_path.replace("/", "___"),
            Path("airgap_bundle/datasets_cache") / dataset_path.replace("/", "__"),
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


DATASET_HELP = (
    "HF dataset ID (default: tatsu-lab/alpaca). Auto-detected schemas:\n"
    "  alpaca   — tatsu-lab/alpaca, bigcode/self-oss-instruct-sc2-exec-filter-50k\n"
    "  sharegpt — teknium/OpenHermes-2.5\n"
    "  messages — HuggingFaceH4/no_robots\n"
    "  dpo      — argilla/distilabel-capybara-dpo-7k-binarized\n"
    "  guanaco  — mlabonne/guanaco-llama2-1k"
)


def validate_dataset_schema(dataset, dataset_name="", n_check=3, logger=None):
    """Verify schema compatibility on the first n_check examples before training starts.

    Detects the schema, extracts prompt+response, and warns if either is empty.
    Returns True if all checked examples are valid, False otherwise.
    """
    import logging
    log = logger or logging.getLogger(__name__)

    n = min(n_check, len(dataset))
    schema = None
    ok = True
    for i in range(n):
        ex = dataset[i]
        schema = _detect_schema(ex)
        prompt, response = _extract_pair(ex)
        if not prompt or not response:
            log.warning(
                "Schema check [%d/%d]: schema=%s — empty %s. "
                "Check field names for dataset '%s'.",
                i + 1, n, schema,
                "prompt" if not prompt else "response",
                dataset_name,
            )
            ok = False

    if ok and schema:
        log.info("Dataset schema OK: %s (%d/%d examples validated)", schema, n, n)
    return ok


def _detect_schema(example):
    """Detect dataset field schema from a single example.

    Supported schemas:
      sharegpt  – teknium/OpenHermes-2.5 and similar; 'conversations' list
                  with {from: human/gpt/system, value: str}
      messages  – HuggingFaceH4/no_robots and ChatML datasets; 'messages' list
                  with {role: user/assistant, content: str}
      dpo       – argilla/distilabel-capybara-dpo-7k-binarized; 'instruction'
                  + 'chosen' list with {role, content}
      guanaco   – mlabonne/guanaco-llama2-1k; 'text' with ### Human: markers
      alpaca    – tatsu-lab/alpaca, bigcode/self-oss-instruct-sc2-exec-filter-50k
                  and any dataset with instruction/prompt + output/response fields
    """
    if "conversations" in example:
        return "sharegpt"
    if "messages" in example and isinstance(example.get("messages"), list):
        return "messages"
    if "chosen" in example and isinstance(example.get("chosen"), list):
        return "dpo"
    if "text" in example and "### Human:" in str(example.get("text", "")):
        return "guanaco"
    return "alpaca"


def _extract_pair(example):
    """Return (prompt, response) strings for any supported schema."""
    schema = _detect_schema(example)

    if schema == "sharegpt":
        prompt = response = ""
        for turn in example["conversations"]:
            role = turn.get("from", turn.get("role", ""))
            value = turn.get("value", turn.get("content", "")).strip()
            if role in ("human", "user") and not prompt:
                prompt = value
            elif role in ("gpt", "assistant") and not response:
                response = value
        return prompt, response

    if schema == "messages":
        prompt = response = ""
        for msg in example["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if role == "user" and not prompt:
                prompt = content
            elif role == "assistant" and not response:
                response = content
        if not prompt:
            prompt = example.get("prompt", "").strip()
        return prompt, response

    if schema == "dpo":
        prompt = example.get("instruction", "").strip()
        response = ""
        for msg in example.get("chosen", []):
            if msg.get("role") == "assistant":
                response = msg.get("content", "").strip()
                break
        return prompt, response

    if schema == "guanaco":
        text = example.get("text", "")
        parts = text.split("### Assistant:", 1)
        response = parts[1].strip() if len(parts) > 1 else ""
        human_parts = parts[0].split("### Human:", 1)
        prompt = human_parts[1].strip() if len(human_parts) > 1 else ""
        return prompt, response

    # alpaca (default)
    prompt = example.get("instruction", example.get("prompt", "")).strip()
    inp = example.get("input", "").strip()
    if inp:
        prompt += "\n\nInput: " + inp
    response = example.get("output", example.get("response", "")).strip()
    return prompt, response


def format_prompt_only(example):
    """Format an example as a prompt string (no response).

    Handles: alpaca, sharegpt, messages (ChatML), dpo, guanaco schemas.
    Used by: distill_sft.py (teacher label generation), distill_minillm.py.
    """
    prompt, _ = _extract_pair(example)
    return prompt + "\n\n### Response:"


def format_prompt_full(example):
    """Format an example as prompt + response (full sequence).

    Handles: alpaca, sharegpt, messages (ChatML), dpo, guanaco schemas.
    Used by: distill_mlx.py (forward KD over complete sequences).
    """
    prompt, response = _extract_pair(example)
    return prompt + "\n\n### Response:\n" + response


def format_multiturn_full(example, max_turns=4):
    """Format a multi-turn conversation as a full ChatML string.

    Renders all turns (up to max_turns user+assistant pairs) using the
    <|im_start|>/<|im_end|> chat template. Falls back to format_prompt_full
    for single-turn schemas (alpaca, dpo, guanaco).

    Used by: distill_mlx.py when --multi_turn_ratio > 0.
    """
    schema = _detect_schema(example)

    if schema == "sharegpt":
        parts = []
        pair_count = 0
        for turn in example.get("conversations", []):
            if pair_count >= max_turns:
                break
            role = turn.get("from", turn.get("role", ""))
            value = turn.get("value", turn.get("content", "")).strip()
            if not value:
                continue
            if role == "system":
                parts.append(f"<|im_start|>system\n{value}<|im_end|>")
            elif role in ("human", "user"):
                parts.append(f"<|im_start|>user\n{value}<|im_end|>")
            elif role in ("gpt", "assistant"):
                parts.append(f"<|im_start|>assistant\n{value}<|im_end|>")
                pair_count += 1
        return "\n".join(parts)

    if schema == "messages":
        parts = []
        pair_count = 0
        for msg in example.get("messages", []):
            if pair_count >= max_turns:
                break
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if not content:
                continue
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
                pair_count += 1
        return "\n".join(parts)

    # Single-turn schemas — fall back to standard formatting
    return format_prompt_full(example)


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
