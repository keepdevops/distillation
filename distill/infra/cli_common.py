"""Shared argparse fragments for training/eval CLIs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def add_hf_dataset_args(
    p: argparse.ArgumentParser,
    *,
    dataset_default: str = "tatsu-lab/alpaca",
) -> None:
    p.add_argument(
        "--dataset",
        type=str,
        default=dataset_default,
        help="HF dataset ID or local path",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HF datasets cache override",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Do not use the network (HF_HUB_OFFLINE / HF_DATASETS_OFFLINE)",
    )


def add_output_dir_arg(
    p: argparse.ArgumentParser,
    *,
    required: bool = True,
    default: str | None = None,
    help_text: str = "Training/output directory",
) -> None:
    p.add_argument(
        "--output_dir",
        type=str,
        required=required,
        default=default,
        help=help_text,
    )


def add_cache_and_offline(p: argparse.ArgumentParser) -> None:
    """--cache_dir and --offline (shared by eval, filtering, and several distill scripts)."""
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument(
        "--offline",
        action="store_true",
        help="Do not use the network (sets HF hub/datasets offline env)",
    )


def apply_offline_env(offline: bool) -> None:
    if offline or os.environ.get("HF_HUB_OFFLINE") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
