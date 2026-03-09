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
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OPEN_MODELS = [
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
]
LLAMA_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-8B-Instruct",
]
BENCH_MODELS = [
    "distilbert-base-uncased",
    "bert-large-uncased",
]
DATASETS = [
    ("tatsu-lab/alpaca", None),
    ("wikitext", "wikitext-2-raw-v1"),
    ("glue", "sst2"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(paths: list[Path], out: Path):
    lines = []
    for p in paths:
        if p.is_file():
            lines.append(f"{sha256(p)}  {p.name}")
    out.write_text("\n".join(lines) + "\n")
    log.info("Checksums written to %s", out)


def verify_checksums(sums_file: Path):
    log.info("Verifying checksums...")
    ok = True
    for line in sums_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split("  ", 1)
        p = sums_file.parent / name
        if not p.exists():
            log.error("  MISSING: %s", name)
            ok = False
        elif sha256(p) != digest:
            log.error("  CORRUPT: %s", name)
            ok = False
        else:
            log.info("  OK: %s", name)
    return ok


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

MLX_CONVERT_MAP = {
    "Qwen/Qwen2-1.5B-Instruct": "qwen2-1.5b-instruct",
    "Qwen/Qwen2-0.5B-Instruct": "qwen2-0.5b-instruct",
    "meta-llama/Llama-3.2-8B-Instruct": "llama-3.2-8b-instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "llama-3.2-1b-instruct",
}


def _convert_mlx(hf_path: str, mlx_path: Path, q_bits: int = 4):
    """Convert a cached HF model to MLX format using mlx_lm.convert."""
    if mlx_path.exists() and any(mlx_path.iterdir()):
        log.info("  MLX already exists, skipping: %s", mlx_path)
        return
    mlx_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", hf_path,
        "--mlx-path", str(mlx_path),
        "--q-bits", str(q_bits),
    ]
    log.info("  Converting to MLX (q%d): %s -> %s", q_bits, hf_path, mlx_path)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.warning("  MLX conversion failed for %s (non-fatal)", hf_path)
    else:
        log.info("  MLX conversion done: %s", mlx_path)


def cmd_prepare(args):
    output = Path(args.output).resolve()
    hf_cache = output / "hf_cache"
    ds_cache = output / "datasets_cache"
    mlx_models = output / "mlx_models"
    hf_cache.mkdir(parents=True, exist_ok=True)
    ds_cache.mkdir(parents=True, exist_ok=True)
    mlx_models.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_DATASETS_CACHE"] = str(ds_cache)

    models = OPEN_MODELS if args.open else LLAMA_MODELS
    if not args.skip_bench:
        models = models + BENCH_MODELS

    # --- models ---
    log.info("=" * 60)
    log.info("STEP 1/3  Caching models -> %s", hf_cache)
    log.info("=" * 60)
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    failed = []
    for name in models:
        log.info("  %s", name)
        try:
            AutoTokenizer.from_pretrained(name, cache_dir=str(hf_cache))
            if any(k in name for k in ("Qwen", "Llama", "llama", "GPT", "gpt")):
                AutoModelForCausalLM.from_pretrained(name, cache_dir=str(hf_cache))
            else:
                AutoModelForSequenceClassification.from_pretrained(name, cache_dir=str(hf_cache))
            log.info("    cached")
        except Exception as e:
            log.warning("    FAILED: %s", e)
            failed.append(name)

    if failed and not args.open:
        log.error("Required models failed — check HF login: %s", failed)
        sys.exit(1)

    # --- MLX conversion (Apple Silicon — one-time, loads instantly offline) ---
    log.info("=" * 60)
    log.info("STEP 1b/3  Converting models to MLX format -> %s", mlx_models)
    log.info("=" * 60)
    causal_models = [m for m in models if any(k in m for k in ("Qwen", "Llama", "llama", "GPT", "gpt"))]
    for name in causal_models:
        short = MLX_CONVERT_MAP.get(name)
        if short is None:
            continue
        # Resolve actual snapshot path from hf_cache
        hf_model_dir = hf_cache / ("models--" + name.replace("/", "--"))
        snapshots = sorted((hf_model_dir / "snapshots").iterdir()) if (hf_model_dir / "snapshots").exists() else []
        if not snapshots:
            log.warning("  No HF snapshot found for %s — skipping MLX conversion", name)
            continue
        snapshot_path = snapshots[-1]  # latest snapshot
        _convert_mlx(str(snapshot_path), mlx_models / short)

    # --- datasets ---
    log.info("=" * 60)
    log.info("STEP 2/3  Caching datasets -> %s", ds_cache)
    log.info("=" * 60)
    from datasets import load_dataset
    disk_paths = {}
    for name, config in DATASETS:
        label = f"{name}" + (f":{config}" if config else "")
        log.info("  %s", label)
        try:
            ds = load_dataset(name, config, cache_dir=str(ds_cache))
            disk_name = name.replace("/", "___") + (f"_{config}" if config else "")
            disk_path = ds_cache / disk_name
            ds.save_to_disk(str(disk_path))
            disk_paths[label] = disk_path
            log.info("    saved -> %s", disk_path)
        except Exception as e:
            log.warning("    skipped: %s", e)

    # --- conda pack ---
    if args.pack_env:
        log.info("=" * 60)
        log.info("STEP 3/3  Packing conda env")
        log.info("=" * 60)
        tarball = output / "distill-offline.tar.gz"
        env_name = os.environ.get("CONDA_DEFAULT_ENV", "distillation_m3")
        log.info("  conda pack -n %s -o %s", env_name, tarball)
        result = subprocess.run(
            ["conda", "pack", "-n", env_name, "-o", str(tarball), "--ignore-editable-packages"],
            capture_output=False,
        )
        if result.returncode != 0:
            log.error("conda pack failed — pack manually and retry without --pack-env")
        else:
            write_checksums([tarball], output / "SHA256SUMS")
    else:
        log.info("STEP 3/3  Skipped conda pack (pass --pack-env to include)")

    # --- summary ---
    log.info("")
    log.info("=" * 60)
    log.info("Bundle ready: %s", output)
    log.info("=" * 60)
    log.info("Transfer to target via USB:")
    log.info("  %s/hf_cache/", output)
    log.info("  %s/datasets_cache/", output)
    if args.pack_env:
        log.info("  %s/distill-offline.tar.gz", output)
        log.info("  %s/SHA256SUMS", output)
    log.info("")
    log.info("On target, run:")
    log.info("  python scripts/airgap.py run --bundle <path> --open%s",
             " --backend mlx" if args.open else "")


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
