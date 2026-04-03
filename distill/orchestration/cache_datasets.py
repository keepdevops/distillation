#!/usr/bin/env python3
"""
Pre-cache Hugging Face datasets for air-gapped transfer.
Run on staging machine with network access.
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DATASETS = [
    ("tatsu-lab/alpaca", None),
    ("glue", "sst2"),
    ("teknium/OpenHermes-2.5", None),
    ("HuggingFaceH4/no_robots", None),
    ("argilla/distilabel-capybara-dpo-7k-binarized", None),
    ("mlabonne/guanaco-llama2-1k", None),
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k", None),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="./datasets_cache")
    p.add_argument("--disk", action="store_true",
                   help="Save load_from_disk copies for air-gapped (--dataset /path/to/alpaca)")
    p.add_argument("--datasets", type=str, nargs="+", default=None, help="e.g. glue:sst2 tatsu-lab/alpaca")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = os.path.abspath(args.output)

    from datasets import load_dataset

    if args.datasets:
        to_load = []
        for d in args.datasets:
            if ":" in d:
                name, config = d.split(":", 1)
                to_load.append((name, config))
            else:
                to_load.append((d, None))
    else:
        to_load = DEFAULT_DATASETS

    for name, config in to_load:
        if not name or not isinstance(name, str):
            logger.warning("Skipping invalid dataset: %r", (name, config))
            continue
        label = f"{name}" + (f" ({config})" if config else "")
        logger.info("Caching %s...", label)
        try:
            ds = load_dataset(name, config, cache_dir=args.output)
            if "train" in ds:
                logger.info("  Train: %s examples", len(ds["train"]))
            else:
                logger.info("  Loaded: %s", list(ds.keys()))
            if args.disk:
                # Use "___" separator to match load_dataset_split offline lookup
                disk_name = (name.replace("/", "___") + (f"_{config}" if config else ""))
                disk_path = os.path.join(args.output, disk_name)
                ds.save_to_disk(disk_path)
                logger.info("  Saved to disk: %s (use --dataset %s)", disk_path, disk_path)
        except Exception as e:
            logger.warning("  Failed %s: %s", label, e)

    logger.info("Datasets cached to %s", args.output)


if __name__ == "__main__":
    main()
