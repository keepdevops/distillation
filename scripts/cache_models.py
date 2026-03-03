#!/usr/bin/env python3
"""
Pre-cache Hugging Face models for air-gapped transfer.
Run on staging machine with network access.
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-8B-Instruct",
    "distilbert-base-uncased",
    "bert-large-uncased",
]

OPEN_MODELS = [
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "distilbert-base-uncased",
    "bert-large-uncased",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="./hf_cache")
    p.add_argument("--open", action="store_true",
                   help="Cache open models (Qwen2) — no Meta license")
    p.add_argument("--models", type=str, nargs="+", default=None)
    args = p.parse_args()

    models = args.models or (OPEN_MODELS if args.open else DEFAULT_MODELS)

    os.makedirs(args.output, exist_ok=True)
    os.environ["HF_HOME"] = os.path.abspath(args.output)

    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    for name in models:
        if not name or not isinstance(name, str):
            logger.warning("Skipping invalid model name: %r", name)
            continue
        logger.info("Caching %s...", name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=args.output)
            if "Llama" in name or "llama" in name or "Causal" in str(name):
                AutoModelForCausalLM.from_pretrained(name, cache_dir=args.output)
            else:
                AutoModelForSequenceClassification.from_pretrained(name, cache_dir=args.output)
            logger.info("  Cached: %s", name)
        except Exception as e:
            logger.warning("  Failed %s: %s", name, e)
    logger.info("Models cached to %s", args.output)


if __name__ == "__main__":
    main()
