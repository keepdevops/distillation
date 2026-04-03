"""Staging-side helpers for airgap bundle preparation (cmd_prepare)."""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

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

MLX_CONVERT_MAP = {
    "Qwen/Qwen2-1.5B-Instruct": "qwen2-1.5b-instruct",
    "Qwen/Qwen2-0.5B-Instruct": "qwen2-0.5b-instruct",
    "meta-llama/Llama-3.2-8B-Instruct": "llama-3.2-8b-instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "llama-3.2-1b-instruct",
}


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


def verify_checksums(sums_file: Path) -> bool:
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
        raise RuntimeError(f"MLX conversion failed for {hf_path} (exit {result.returncode})")
    log.info("  MLX conversion done: %s", mlx_path)


def cmd_prepare(args) -> None:
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

    log.info("=" * 60)
    log.info("STEP 1b/3  Converting models to MLX format -> %s", mlx_models)
    log.info("=" * 60)
    causal_models = [m for m in models if any(k in m for k in ("Qwen", "Llama", "llama", "GPT", "gpt"))]
    for name in causal_models:
        short = MLX_CONVERT_MAP.get(name)
        if short is None:
            continue
        hf_model_dir = hf_cache / ("models--" + name.replace("/", "--"))
        snapshots = sorted((hf_model_dir / "snapshots").iterdir()) if (hf_model_dir / "snapshots").exists() else []
        if not snapshots:
            log.warning("  No HF snapshot found for %s — skipping MLX conversion", name)
            continue
        _convert_mlx(str(snapshots[-1]), mlx_models / short)

    log.info("=" * 60)
    log.info("STEP 2/3  Caching datasets -> %s", ds_cache)
    log.info("=" * 60)
    from datasets import load_dataset
    for name, config in DATASETS:
        label = f"{name}" + (f":{config}" if config else "")
        log.info("  %s", label)
        try:
            ds = load_dataset(name, config, cache_dir=str(ds_cache))
            disk_name = name.replace("/", "___") + (f"_{config}" if config else "")
            ds.save_to_disk(str(ds_cache / disk_name))
            log.info("    saved -> %s", ds_cache / disk_name)
        except Exception as e:
            log.warning("    skipped: %s", e)

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
    log.info("  python -m distill.airgap run --bundle <path> --open%s",
             " --backend mlx" if args.open else "")
