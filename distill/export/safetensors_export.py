"""Safetensors + HuggingFace Hub export.

Converts a trained checkpoint to safetensors format and optionally
pushes to a private HF Hub repository.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def export_safetensors(
    model_path: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_repo_id: str = "",
    hub_token: str | None = None,
    private: bool = True,
) -> dict[str, Any]:
    """Export model to safetensors format (and optionally push to HF Hub).

    Returns a result dict with keys: output_path, pushed, error.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Loading model from %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cpu", low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info("Saving safetensors to %s", out)
        model.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
    except Exception as exc:
        logger.error("safetensors export failed: %s", exc)
        return {"output_path": "", "pushed": False, "error": str(exc)}

    pushed = False
    if push_to_hub and hub_repo_id:
        pushed = _push_to_hub(out, hub_repo_id, hub_token, private)

    return {"output_path": str(out), "pushed": pushed, "error": ""}


def _push_to_hub(
    model_dir: Path,
    repo_id: str,
    token: str | None,
    private: bool,
) -> bool:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        logger.info("Pushed to HF Hub: %s", repo_id)
        return True
    except Exception as exc:
        logger.error("HF Hub push failed: %s", exc)
        return False


def model_card_markdown(
    model_id: str,
    teacher: str,
    dataset: str,
    backend: str,
    metrics: dict[str, Any],
) -> str:
    """Generate a minimal HF model card."""
    ppl = metrics.get("perplexity", "N/A")
    quality = metrics.get("quality_score", "N/A")
    return f"""---
language: en
tags:
  - distilled
  - knowledge-distillation
  - lora
base_model: {teacher}
---

# {model_id}

Distilled from **{teacher}** using [Wow Sausage Maker](https://github.com/keepdevops/distillation).

## Training Details
- **Backend:** {backend}
- **Dataset:** {dataset}
- **Perplexity:** {ppl}
- **Quality Score:** {quality}

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{model_id}")
tokenizer = AutoTokenizer.from_pretrained("{model_id}")
```
"""
