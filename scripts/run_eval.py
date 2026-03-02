#!/usr/bin/env python3
"""
Standalone eval: compute cross-entropy loss on the validation split and
append eval_loss + perplexity to {output_dir}/metrics.jsonl.

Usage:
    python scripts/run_eval.py ./distilled-minillm
    python scripts/run_eval.py ./distilled-minillm --checkpoint ./distilled-minillm/checkpoint-80
    python scripts/run_eval.py ./distilled-minillm --step 90
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPEN_STUDENT = "Qwen/Qwen2-0.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description="Standalone eval for distilled model")
    p.add_argument("output_dir", type=str, help="Training output dir (metrics.jsonl written here)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Model checkpoint dir to eval (default: output_dir itself)")
    p.add_argument("--student", type=str, default=OPEN_STUDENT,
                   help="Base model id (for tokenizer if checkpoint has none)")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--val_size", type=float, default=0.02)
    p.add_argument("--max_val_samples", type=int, default=200,
                   help="Cap validation samples to keep eval fast")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--step", type=int, default=None,
                   help="Step number to record in metrics.jsonl (default: auto-detect from checkpoint)")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


def format_example(example):
    prompt = example.get("instruction", example.get("prompt", ""))
    if "input" in example and example["input"]:
        prompt += "\n\nInput: " + example["input"]
    prompt += "\n\n### Response:"
    response = example.get("output", example.get("response", ""))
    return {"text": prompt + " " + response}


def detect_step(checkpoint_dir):
    """Infer step number from checkpoint dir name (e.g. checkpoint-80 → 80)."""
    name = Path(checkpoint_dir).name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[-1])
        except ValueError:
            pass
    # Fall back to last step in metrics.jsonl
    return None


def last_step_in_jsonl(jsonl_path):
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


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def eval_loss(model, tokenizer, texts, max_length, batch_size, device):
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
        # Mask padding tokens in loss
        labels[attention_mask == 0] = -100

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # out.loss is mean over non-masked tokens in the batch
        n_tokens = (labels != -100).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return None
    return total_loss / total_tokens


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else output_dir
    jsonl_path = output_dir / "metrics.jsonl"

    offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    cache_dir = os.environ.get("HF_HOME") or args.cache_dir

    # Determine step
    step = args.step
    if step is None:
        step = detect_step(checkpoint_dir)
    if step is None:
        step = last_step_in_jsonl(jsonl_path)
    logger.info("Recording eval at step %d", step)

    # Load tokenizer — prefer checkpoint dir, fall back to base model
    tok_dir = str(checkpoint_dir) if (checkpoint_dir / "tokenizer_config.json").exists() else args.student
    logger.info("Loading tokenizer from %s", tok_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, cache_dir=cache_dir, local_files_only=offline)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model — handle both full models and LoRA adapter checkpoints
    logger.info("Loading model from %s", checkpoint_dir)
    device = get_device()
    is_adapter = (checkpoint_dir / "adapter_config.json").exists()
    if is_adapter:
        logger.info("Detected LoRA adapter checkpoint — loading base + adapter")
        # Read adapter_config to find base model id
        with open(checkpoint_dir / "adapter_config.json") as f:
            adapter_cfg = json.load(f)
        base_model_id = adapter_cfg.get("base_model_name_or_path", args.student)
        logger.info("Base model: %s", base_model_id)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=offline,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(checkpoint_dir))
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=offline,
        )
    model.to(device)

    # Load dataset
    logger.info("Loading dataset %s", args.dataset)
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    if Path(args.dataset).exists():
        data = load_from_disk(args.dataset)
        dataset = data["train"] if isinstance(data, dict) and "train" in data else data
    else:
        dataset = load_dataset(args.dataset, split="train", cache_dir=ds_cache)

    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    split = dataset.train_test_split(test_size=args.val_size, seed=42)
    val_ds = split["test"]

    if args.max_val_samples and len(val_ds) > args.max_val_samples:
        val_ds = val_ds.select(range(args.max_val_samples))
    logger.info("Validation samples: %d", len(val_ds))

    texts = val_ds["text"]
    loss = eval_loss(model, tokenizer, texts, args.max_length, args.batch_size, device)
    if loss is None:
        logger.error("Could not compute eval loss (no tokens)")
        raise SystemExit(1)

    perplexity = math.exp(min(loss, 20))
    logger.info("step=%d  eval_loss=%.4f  perplexity=%.2f", step, loss, perplexity)

    # Append to metrics.jsonl
    row = {"step": step, "eval_loss": loss, "eval_perplexity": perplexity}
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info("Appended to %s", jsonl_path)


if __name__ == "__main__":
    main()
