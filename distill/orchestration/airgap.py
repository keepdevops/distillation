#!/usr/bin/env python3
"""
Air-gapped distillation — single entry point.

STAGING (online, run once):
  python scripts/airgap.py prepare --open --output ./airgap_bundle
  python scripts/airgap.py prepare --open --output ./airgap_bundle --pack-env

TARGET (offline):
  python scripts/airgap.py run --bundle ./airgap_bundle --open
  python scripts/airgap.py run --bundle ./airgap_bundle --open --backend mlx --export gguf
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from .airgap_prepare import (  # noqa: F401 — re-exported for callers
    cmd_prepare, verify_checksums, write_checksums, sha256,
    OPEN_MODELS, LLAMA_MODELS, BENCH_MODELS, DATASETS,
)


# ---------------------------------------------------------------------------
# run (offline target)
# ---------------------------------------------------------------------------

def cmd_run(args):
    bundle = Path(args.bundle).resolve()
    hf_cache = bundle / "hf_cache"
    ds_cache = bundle / "datasets_cache"

    # Verify bundle exists
    if not hf_cache.exists() or not ds_cache.exists():
        log.error("Bundle not found at %s — run 'prepare' on staging first", bundle)
        sys.exit(1)

    # Verify checksums if present
    sums = bundle / "SHA256SUMS"
    if sums.exists():
        if not verify_checksums(sums):
            log.error("Checksum verification failed — re-transfer bundle")
            sys.exit(1)
    else:
        log.warning("No SHA256SUMS found — skipping integrity check")

    # Set offline env vars
    os.environ.update({
        "HF_HOME": str(hf_cache),
        "HF_DATASETS_CACHE": str(ds_cache),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    log.info("Offline env set: HF_HOME=%s", hf_cache)

    # Verify MPS
    try:
        import torch
        mps_ok = torch.backends.mps.is_available()
        log.info("MPS available: %s", mps_ok)
    except ImportError:
        log.warning("torch not importable — check env")

    # Find alpaca disk path
    alpaca_disk = ds_cache / "tatsu-lab___alpaca"
    dataset_arg = ["--dataset", str(alpaca_disk)] if alpaca_disk.exists() else []

    # Build agent command
    agent = Path(__file__).parent / "run_distillation_agent.py"
    cmd = [sys.executable, str(agent), "--offline"]
    if args.open:
        cmd.append("--open")
    cmd += ["--backend", args.backend]
    cmd += ["--export", args.export]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.max_samples:
        cmd += ["--max_samples", str(args.max_samples)]
    if args.log_experiment:
        cmd.append("--log_experiment")
    if args.watchdog:
        cmd.append("--watchdog")
    if args.skip_eval:
        cmd.append("--skip_eval")
    cmd += dataset_arg

    log.info("=" * 60)
    log.info("Running: %s", " ".join(cmd))
    log.info("=" * 60)
    os.execv(sys.executable, cmd)  # replace process — streams pass through cleanly


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Air-gapped distillation — one script for staging and offline target",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # prepare
    prep = sub.add_parser("prepare", help="Staging machine (online): cache models + datasets")
    prep.add_argument("--output", default="./airgap_bundle",
                      help="Directory for cached models, datasets, env tarball")
    prep.add_argument("--open", action="store_true",
                      help="Use Qwen2 (no HF login); otherwise Llama (requires login + license)")
    prep.add_argument("--pack-env", action="store_true",
                      help="Run conda pack on current env and add to bundle")
    prep.add_argument("--skip-bench", action="store_true",
                      help="Skip BERT/DistilBERT benchmark models")

    # run
    run = sub.add_parser("run", help="Target machine (offline): set env + run distillation")
    run.add_argument("--bundle", default="./airgap_bundle",
                     help="Path to bundle directory created by 'prepare'")
    run.add_argument("--open", action="store_true", help="Use open models (Qwen2)")
    run.add_argument("--backend", default="mlx", choices=["mlx", "pytorch", "unsloth"],
                     help="Training backend (default: mlx)")
    run.add_argument("--export", default="gguf", choices=["gguf", "coreml", "all", "none"],
                     help="Export format after training (default: gguf)")
    run.add_argument("--epochs", type=int, default=None)
    run.add_argument("--max-samples", dest="max_samples", type=int, default=None,
                     help="Limit training samples (useful for quick test)")
    run.add_argument("--log-experiment", dest="log_experiment", action="store_true")
    run.add_argument("--watchdog", action="store_true", help="Enable plateau detection watchdog")
    run.add_argument("--skip-eval", dest="skip_eval", action="store_true")

    args = p.parse_args()
    if args.cmd == "prepare":
        cmd_prepare(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
