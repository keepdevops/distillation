"""AWQ and GPTQ quantization exporters.

AWQ (Activation-aware Weight Quantization) — better quality than GPTQ at 4-bit.
GPTQ — classic 4-bit, widely supported (exllama, vLLM, text-generation-inference).

Both require GPU (CUDA) for calibration. These exports are skipped gracefully
on Apple Silicon with a clear warning pointing to MLX/GGUF alternatives.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _require_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def export_awq(
    model_path: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
) -> dict[str, Any]:
    """Quantize a model to AWQ format.

    Requires: pip install autoawq
    GPU (CUDA) required for calibration.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not _require_cuda():
        msg = ("AWQ calibration requires a CUDA GPU. "
               "On Apple Silicon, use MLX (mlx_q4/) or GGUF instead.")
        logger.warning(msg)
        return {"output_path": "", "format": "awq", "error": msg}

    try:
        from awq import AutoAWQForCausalLM  # type: ignore[import]
        from transformers import AutoTokenizer

        logger.info("Loading model for AWQ quantization: %s", model_path)
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        quant_config = {
            "zero_point": zero_point,
            "q_group_size": group_size,
            "w_bit": bits,
            "version": "GEMM",
        }
        logger.info("Running AWQ calibration (bits=%d, group=%d)...", bits, group_size)
        model.quantize(tokenizer, quant_config=quant_config)
        model.save_quantized(str(out))
        tokenizer.save_pretrained(out)
        logger.info("AWQ export complete: %s", out)
        return {"output_path": str(out), "format": "awq", "error": ""}

    except ImportError:
        msg = "autoawq not installed. Run: pip install autoawq"
        logger.error(msg)
        return {"output_path": "", "format": "awq", "error": msg}
    except Exception as exc:
        logger.error("AWQ export failed: %s", exc)
        return {"output_path": "", "format": "awq", "error": str(exc)}


def export_gptq(
    model_path: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    dataset: str = "c4",
) -> dict[str, Any]:
    """Quantize a model to GPTQ format.

    Requires: pip install auto-gptq
    GPU (CUDA) required for calibration.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not _require_cuda():
        msg = ("GPTQ calibration requires CUDA. "
               "On Apple Silicon, use GGUF (Q4_K_M) or MLX instead.")
        logger.warning(msg)
        return {"output_path": "", "format": "gptq", "error": msg}

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore[import]
        from transformers import AutoTokenizer

        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=False,
        )
        logger.info("Loading model for GPTQ quantization...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)

        logger.info("GPTQ calibration on dataset=%s ...", dataset)
        examples = _get_calibration_examples(tokenizer, dataset, n=128)
        model.quantize(examples)
        model.save_quantized(str(out), use_safetensors=True)
        tokenizer.save_pretrained(out)
        logger.info("GPTQ export complete: %s", out)
        return {"output_path": str(out), "format": "gptq", "error": ""}

    except ImportError:
        msg = "auto-gptq not installed. Run: pip install auto-gptq"
        logger.error(msg)
        return {"output_path": "", "format": "gptq", "error": msg}
    except Exception as exc:
        logger.error("GPTQ export failed: %s", exc)
        return {"output_path": "", "format": "gptq", "error": str(exc)}


def _get_calibration_examples(tokenizer: Any, dataset: str, n: int = 128) -> list[dict]:
    """Load a small calibration set."""
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset(dataset, "en", split=f"train[:{n}]", trust_remote_code=True)
    text_field = "text" if "text" in ds.column_names else ds.column_names[0]
    examples = []
    for row in ds:
        enc = tokenizer(str(row[text_field]), return_tensors="pt", truncation=True,
                        max_length=512)
        examples.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return examples
