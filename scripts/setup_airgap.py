#!/usr/bin/env python3
"""
All-in-one script to cache everything needed for air-gapped distillation.
Run this once with internet access, then use --offline flag for all operations.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cache_models(cache_dir, open_models=True):
    """Cache teacher and student models."""
    logger.info("=" * 70)
    logger.info("STEP 1/3: Caching Models")
    logger.info("=" * 70)

    if open_models:
        models = [
            "Qwen/Qwen2-0.5B-Instruct",   # Student
            "Qwen/Qwen2-1.5B-Instruct",   # Teacher
        ]
    else:
        models = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-8B-Instruct",
        ]

    # Optional benchmark models
    benchmark_models = [
        "distilbert-base-uncased",
        "bert-large-uncased",
    ]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    for model_id in models:
        logger.info("Caching %s...", model_id)
        try:
            AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
            AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
            logger.info("  ✓ Cached: %s", model_id)
        except Exception as e:
            logger.error("  ✗ Failed %s: %s", model_id, e)
            return False

    logger.info("\nOptional benchmark models (can skip if size constrained):")
    for model_id in benchmark_models:
        try:
            logger.info("Caching %s...", model_id)
            from transformers import AutoModelForSequenceClassification
            AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
            AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir)
            logger.info("  ✓ Cached: %s", model_id)
        except Exception as e:
            logger.warning("  ⚠ Skipped %s: %s", model_id, e)

    return True


def cache_datasets(cache_dir, save_disk=False):
    """Cache training and evaluation datasets."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2/3: Caching Datasets")
    logger.info("=" * 70)

    datasets_to_cache = [
        ("tatsu-lab/alpaca", None),          # Training data
        ("wikitext", "wikitext-2-raw-v1"),   # Benchmark (WikiText-2)
        ("glue", "sst2"),                    # Benchmark (optional)
    ]

    from datasets import load_dataset

    for name, config in datasets_to_cache:
        label = f"{name}" + (f":{config}" if config else "")
        logger.info("Caching %s...", label)
        try:
            ds = load_dataset(name, config, cache_dir=cache_dir)
            if "train" in ds:
                logger.info("  ✓ Cached %s examples", len(ds["train"]))
            else:
                logger.info("  ✓ Cached splits: %s", list(ds.keys()))

            if save_disk:
                disk_name = name.replace("/", "_") + (f"_{config}" if config else "")
                disk_path = Path(cache_dir) / disk_name
                ds.save_to_disk(str(disk_path))
                logger.info("  ✓ Saved to disk: %s", disk_path)
        except Exception as e:
            if "wikitext" in name or "glue" in name:
                logger.warning("  ⚠ Skipped benchmark dataset %s: %s", label, e)
            else:
                logger.error("  ✗ Failed required dataset %s: %s", label, e)
                return False

    return True


def verify_cache(cache_dir):
    """Verify cached models and datasets."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3/3: Verifying Cache")
    logger.info("=" * 70)

    hub_dir = Path(cache_dir) / "hub"
    datasets_dir = Path(cache_dir).parent / "datasets"

    # Check models
    required_models = ["Qwen--Qwen2-0.5B-Instruct", "Qwen--Qwen2-1.5B-Instruct"]
    found_models = []

    if hub_dir.exists():
        for model in required_models:
            model_dirs = list(hub_dir.glob(f"models--{model}*"))
            if model_dirs:
                found_models.append(model)
                logger.info("  ✓ Found model: %s", model.replace("--", "/"))
            else:
                logger.error("  ✗ Missing model: %s", model.replace("--", "/"))
    else:
        logger.error("  ✗ Hub directory not found: %s", hub_dir)
        return False

    # Check datasets
    if datasets_dir.exists():
        alpaca_dirs = list(datasets_dir.glob("*alpaca*"))
        if alpaca_dirs:
            logger.info("  ✓ Found dataset: tatsu-lab/alpaca")
        else:
            logger.warning("  ⚠ Dataset not found (may be in cache with different name)")

    if len(found_models) == len(required_models):
        logger.info("\n✓ Cache verification PASSED")
        return True
    else:
        logger.error("\n✗ Cache verification FAILED")
        return False


def show_next_steps():
    """Print next steps after caching."""
    logger.info("\n" + "=" * 70)
    logger.info("SETUP COMPLETE - Next Steps")
    logger.info("=" * 70)
    logger.info("""
1. Set environment variables for guaranteed offline mode:

   export HF_HUB_OFFLINE=1
   export HF_DATASETS_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1

   Or add to ~/.zshrc:
   echo 'export HF_HUB_OFFLINE=1' >> ~/.zshrc
   echo 'export HF_DATASETS_OFFLINE=1' >> ~/.zshrc
   source ~/.zshrc

2. ALWAYS use --offline flag in all commands:

   python scripts/run_distillation_agent.py \\
       --open --offline \\
       --epochs 2 \\
       --export gguf \\
       --log_experiment

3. Test offline mode:

   python scripts/run_distillation_agent.py \\
       --open --offline \\
       --epochs 1 --max_samples 10 \\
       --skip_eval --export none

4. Verify no network activity in logs (should see no download warnings)

5. For full guide, see: AIRGAP_SETUP.md

You can now disconnect from the network! 🔒
""")


def main():
    parser = argparse.ArgumentParser(
        description="Setup air-gapped distillation environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cache to default HF location
  python scripts/setup_airgap.py --open

  # Cache to custom location for transfer
  python scripts/setup_airgap.py --open --output ./airgap_cache

  # Cache with disk copies for manual transfer
  python scripts/setup_airgap.py --open --output ./airgap_cache --disk
        """
    )
    parser.add_argument("--output", type=str, default=None,
                       help="Cache directory (default: ~/.cache/huggingface)")
    parser.add_argument("--open", action="store_true",
                       help="Cache open models (Qwen2, no HF login required)")
    parser.add_argument("--disk", action="store_true",
                       help="Save datasets to disk for manual transfer")
    parser.add_argument("--skip-verify", action="store_true",
                       help="Skip verification step")
    args = parser.parse_args()

    # Determine cache directory
    if args.output:
        cache_dir = os.path.abspath(args.output)
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = cache_dir
    else:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        os.environ["HF_HOME"] = cache_dir

    logger.info("Caching to: %s", cache_dir)
    logger.info("Models: %s", "Qwen2 (open)" if args.open else "Llama (requires HF login)")
    logger.info("")

    # Run caching steps
    if not cache_models(cache_dir, open_models=args.open):
        logger.error("Model caching failed!")
        sys.exit(1)

    if not cache_datasets(cache_dir, save_disk=args.disk):
        logger.error("Dataset caching failed!")
        sys.exit(1)

    # Verify
    if not args.skip_verify:
        if not verify_cache(cache_dir):
            logger.error("Cache verification failed!")
            sys.exit(1)

    # Show next steps
    show_next_steps()

    logger.info("\n✓ Air-gapped setup complete!")


if __name__ == "__main__":
    main()
