#!/usr/bin/env python3
"""
Autonomous distillation agent: runs the full pipeline (distill → export) end-to-end.

Use for headless runs, cron, or LaunchAgent. Supports --watchdog for plateau detection
and --offline for air-gapped.

Usage:
  python scripts/run_distillation_agent.py --open
  python scripts/run_distillation_agent.py --open --watchdog --export gguf
  python scripts/run_distillation_agent.py --backend mlx --export all --open
  python scripts/run_distillation_agent.py --backend unsloth --export coreml --open
  python scripts/run_distillation_agent.py --config configs/agent_config.json
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_cmd(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    """Run command with live stdout streaming; raise on non-zero exit."""
    LOG.info("Running: %s", " ".join(cmd))
    env = env or os.environ.copy()
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            LOG.info("[subprocess] %s", line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def find_llama_cpp(project_root: Path) -> Path | None:
    candidates = [
        Path("/Users/Shared/models/llama.cpp"),
        project_root / "llama.cpp",
        project_root.parent / "llama.cpp",
    ]
    for candidate in candidates:
        if (candidate / "convert_hf_to_gguf.py").exists():
            return candidate
    return None


def _copy_hf_config_files(model_id: str, dest_dir: Path) -> None:
    """Copy config.json and tokenizer files from HF cache to dest_dir."""
    import shutil
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_name = "models--" + model_id.replace("/", "--")
    snapshots_dir = Path(hf_home) / "hub" / cache_name / "snapshots"
    config_src_dir = None
    if snapshots_dir.exists():
        for snap in sorted(snapshots_dir.iterdir()):
            if (snap / "config.json").exists():
                config_src_dir = snap
                break
    if config_src_dir is None:
        raise RuntimeError(
            f"Could not find cached HF config for {model_id}. "
            "Run with network access first to download the model."
        )
    for fname in [
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt",
    ]:
        src = config_src_dir / fname
        if src.exists():
            shutil.copy2(src, dest_dir / fname)
    LOG.info("Copied HF config files from %s", config_src_dir)


def _mlx_weights_to_hf_dir(output_dir: Path) -> Path:
    """Convert MLX .npz weights (with LoRA) into a HF-compatible directory.

    Merges LoRA adapters into the base weights, then saves as safetensors.
    The resulting directory can be passed to convert_hf_to_gguf.py.

    LoRA naming convention from mlx_lm LoRALinear:
      base weight : <layer>.linear.weight
      LoRA A      : <layer>.lora_a.weight  shape (r, in_features)
      LoRA B      : <layer>.lora_b.weight  shape (out_features, r)
      merge       : W + scale * (B @ A)
    where scale = alpha / r = 20.0 / lora_r  (standard LoRA, mlx_lm convention).
    """
    import numpy as np

    with open(output_dir / "distill_config.json") as f:
        dcfg = json.load(f)
    student_id = dcfg["student"]
    lora_r = int(dcfg.get("lora_r", 16))
    lora_scale = 20.0 / lora_r  # alpha=20 (hardcoded in distill_mlx.py), scale = alpha/r

    npz_path = output_dir / "mlx_student_weights.npz"
    LOG.info("Loading MLX weights from %s", npz_path)
    try:
        import mlx.core as mx
        raw = mx.load(str(npz_path))
        weights = {k: np.array(v.astype(mx.float32)) for k, v in raw.items()}
    except Exception:
        weights = {k: v.astype(np.float32) for k, v in np.load(str(npz_path)).items()}

    all_keys = set(weights)
    hf_weights: dict[str, np.ndarray] = {}
    processed: set[str] = set()

    for key in sorted(all_keys):
        if key in processed:
            continue

        # Skip raw LoRA parameter tensors (handled during base weight merge below).
        # mlx_lm stores them as <layer>.lora_a and <layer>.lora_b (raw mx.array,
        # NOT as nn.Linear sub-modules, so no trailing .weight suffix).
        if key.endswith(".lora_a") or key.endswith(".lora_b"):
            processed.add(key)
            continue

        if key.endswith(".linear.weight"):
            # LoRA-wrapped linear — merge adapters into base weight.
            # mlx_lm LoRALinear stores:
            #   linear.weight : (out, in)
            #   lora_a        : (in, r)   — already transposed for x @ lora_a @ lora_b
            #   lora_b        : (r, out)
            # Merged: W + scale * lora_b.T @ lora_a.T  →  (out, in)
            prefix = key[: -len(".linear.weight")]
            a_key = f"{prefix}.lora_a"
            b_key = f"{prefix}.lora_b"
            hf_key = f"{prefix}.weight"
            W = weights[key]
            if a_key in all_keys and b_key in all_keys:
                A = weights[a_key]  # (in, r)
                B = weights[b_key]  # (r, out)
                hf_weights[hf_key] = W + lora_scale * (B.T @ A.T)
                processed.update({key, a_key, b_key})
            else:
                hf_weights[hf_key] = W
                processed.add(key)
        elif key.endswith(".linear.bias"):
            prefix = key[: -len(".linear.bias")]
            hf_weights[f"{prefix}.bias"] = weights[key]
            processed.add(key)
        else:
            hf_weights[key] = weights[key]
            processed.add(key)

    hf_dir = output_dir / "_hf_export"
    hf_dir.mkdir(exist_ok=True)

    try:
        from safetensors.numpy import save_file
        save_file(
            {k: v.astype(np.float16) for k, v in hf_weights.items()},
            str(hf_dir / "model.safetensors"),
        )
    except ImportError:
        import torch
        torch.save(
            {k: torch.tensor(v) for k, v in hf_weights.items()},
            str(hf_dir / "pytorch_model.bin"),
        )

    _copy_hf_config_files(student_id, hf_dir)
    LOG.info("MLX→HF conversion complete: %s (%d tensors)", hf_dir, len(hf_weights))
    return hf_dir


def export_gguf(output_dir: Path, project_root: Path, outtype: str) -> None:
    """Export trained model to GGUF using llama.cpp.

    If output_dir contains MLX weights (.npz) instead of a HF model, the
    LoRA adapters are merged and a temporary HF-compatible directory is
    created first, then passed to convert_hf_to_gguf.py.
    """
    llama_cpp = find_llama_cpp(project_root)
    if not llama_cpp:
        LOG.warning("llama.cpp not found; skipping GGUF export")
        return

    convert_dir = output_dir
    if not (output_dir / "config.json").exists() and (output_dir / "mlx_student_weights.npz").exists():
        LOG.info("MLX output detected — converting to HF format before GGUF export...")
        convert_dir = _mlx_weights_to_hf_dir(output_dir)

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    out_name = output_dir.name + f"-{outtype}.gguf"
    # Save GGUF alongside other models in /Users/Shared/models
    gguf_dir = Path("/Users/Shared/models")
    out_file = gguf_dir / out_name
    run_cmd(
        [
            sys.executable,
            str(convert_script),
            str(convert_dir),
            "--outfile",
            str(out_file),
            "--outtype",
            outtype,
        ],
        project_root,
    )
    LOG.info("GGUF saved: %s", out_file)
    LOG.info("Serve with: cd %s && ./build/bin/llama-server -m %s", llama_cpp, out_file)


def export_coreml(output_dir: Path, project_root: Path, quantize: str | None) -> None:
    """Export model to CoreML .mlpackage."""
    cmd = [
        sys.executable,
        "scripts/export_coreml.py",
        "--model_dir",
        str(output_dir),
        "--output_dir",
        str(output_dir),
    ]
    if quantize and quantize in ("int4", "int8", "float16"):
        cmd += ["--quantize", quantize]
    run_cmd(cmd, project_root)


def export_mlx_quant(output_dir: Path, project_root: Path, q_bits: int) -> None:
    """Quantize a HF-format model directory via mlx_lm.convert.

    output_dir must be a HuggingFace-format directory (config.json + weights),
    which is true for both PyTorch and Unsloth training outputs.
    """
    try:
        import shutil
        from mlx_lm import convert as mlx_convert
        quant_dir = output_dir / f"mlx_q{q_bits}"
        if quant_dir.exists():
            LOG.info("Removing existing quantized dir: %s", quant_dir)
            shutil.rmtree(quant_dir)
        LOG.info("MLX quantization → %s", quant_dir)
        mlx_convert(str(output_dir), quantize=True, q_bits=q_bits, mlx_path=str(quant_dir))
        LOG.info("MLX quantized model saved: %s", quant_dir)
    except ImportError:
        LOG.warning("mlx_lm not installed; skipping MLX quantization export")
    except Exception as e:
        LOG.warning("MLX quantization failed (non-fatal): %s", e)


def _build_distill_cmd(args, output_dir: Path,
                       sft_checkpoint: str | None,
                       dataset_override: str | None,
                       trial_seed: int) -> list[str]:
    """Return the distillation subprocess command for the given config."""
    if args.backend == "pytorch":
        cmd = [
            sys.executable, "scripts/distill_minillm.py",
            "--output_dir", str(output_dir),
            "--epochs", str(args.epochs),
            "--max_samples", str(args.max_samples),
            "--minillm_temp", str(args.temperature),
            "--lora_r", str(args.lora_r),
            "--seed", str(trial_seed),
        ]
        if args.open:
            cmd.append("--open")
        if args.offline:
            cmd.append("--offline")
        if args.watchdog:
            cmd.append("--watchdog")
        if sft_checkpoint:
            cmd += ["--student", sft_checkpoint]
        _ds = dataset_override or getattr(args, "dataset", None)
        if _ds:
            cmd += ["--dataset", _ds]

    elif args.backend == "mlx":
        cmd = [
            sys.executable, "scripts/distill_mlx.py",
            "--output_dir", str(output_dir),
            "--epochs", str(args.epochs),
            "--max_samples", str(args.max_samples),
            "--kd_temp", str(args.temperature),
            "--lora_r", str(args.lora_r),
            "--q_bits", str(args.q_bits),
            "--seed", str(trial_seed),
        ]
        _ds = dataset_override or getattr(args, "dataset", None)
        if _ds:
            cmd += ["--dataset", _ds]
        if args.open:
            cmd.append("--open")
        if args.offline:
            cmd.append("--offline")
        if args.watchdog:
            cmd.append("--watchdog")
        if args.export not in ("mlx", "all"):
            cmd.append("--no_export")

    else:  # unsloth
        cmd = [
            sys.executable, "scripts/distill_unsloth.py",
            "--output_dir", str(output_dir),
            "--epochs", str(args.epochs),
            "--max_samples", str(args.max_samples),
            "--kd_temp", str(args.temperature),
            "--lora_r", str(args.lora_r),
            "--q_bits", str(args.q_bits),
            "--seed", str(trial_seed),
        ]
        _ds = dataset_override or getattr(args, "dataset", None)
        if _ds:
            cmd += ["--dataset", _ds]
        if args.open:
            cmd.append("--open")
        if args.offline:
            cmd.append("--offline")
        if args.watchdog:
            cmd.append("--watchdog")

    return cmd


def main():
    ap = argparse.ArgumentParser(description="Autonomous distillation agent")
    ap.add_argument("--config", type=str, help="JSON config (overrides CLI)")
    ap.add_argument("--output_dir", type=str, default="./distilled-minillm")
    ap.add_argument("--open", action="store_true", help="Use Qwen2 open models (no HF login)")
    ap.add_argument("--offline", action="store_true", help="Air-gapped: local cache only")
    ap.add_argument("--watchdog", action="store_true", help="Enable pause.flag / plateau detection")
    ap.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        choices=["pytorch", "mlx", "unsloth"],
        help="Training backend (default: pytorch — existing behavior unchanged)",
    )
    ap.add_argument(
        "--export",
        type=str,
        default="gguf",
        choices=["gguf", "coreml", "mlx", "all", "none"],
        help="Export format after distillation (default: gguf)",
    )
    # Legacy alias kept for backwards compat
    ap.add_argument("--export-gguf", action="store_true",
                    help="[legacy] Equivalent to --export gguf")
    ap.add_argument("--outtype", type=str, default="f16",
                    help="GGUF quantization type (f16, q8_0, q4_K_M)")
    ap.add_argument("--q_bits", type=int, default=4, choices=[4, 8],
                    help="MLX quantization bits (4 or 8)")
    ap.add_argument("--coreml_quantize", type=str, default=None,
                    choices=["int4", "int8", "float16"],
                    help="CoreML post-training quantization type")
    ap.add_argument("--dataset", type=str, default="tatsu-lab/alpaca",
                    help="HF dataset ID or local path (default: tatsu-lab/alpaca). "
                         "Better options: HuggingFaceH4/no_robots, teknium/OpenHermes-2.5, "
                         "allenai/tulu-3-sft-mixture, Open-Orca/OpenOrca")
    # ── Dataset filtering ─────────────────────────────────────────────────────
    ap.add_argument("--filter", action="store_true",
                    help="Pre-filter dataset with filter_dataset.py before distillation")
    ap.add_argument("--filter_target", type=int, default=10000,
                    help="Keep top-N samples after filtering (default: 10000)")
    ap.add_argument("--filter_min_response_words", type=int, default=30,
                    help="Min response word count for filter (default: 30)")
    ap.add_argument("--filter_min_distinct2", type=float, default=0.40,
                    help="Min distinct-2 score for filter (default: 0.40)")
    ap.add_argument("--filter_jaccard", type=float, default=0.55,
                    help="Jaccard similarity threshold for near-dedup (default: 0.55)")
    ap.add_argument("--filter_minhash", action="store_true",
                    help="Use MinHash LSH for global dedup in filter (requires datasketch)")
    ap.add_argument("--filter_teacher_score", action="store_true",
                    help="Re-rank filtered candidates by teacher log-probability")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="KD temperature (default 1.0)")
    ap.add_argument("--lora_r", type=int, default=16,
                    help="LoRA rank (default: 16). Higher ranks increase memory; "
                         "64 risks OOM on MLX with full-vocab logit tensors.")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--skip_eval", action="store_true",
                    help="Skip post-distillation eval steps (run_eval + eval_quality)")
    ap.add_argument("--compare_teacher", action="store_true",
                    help="Include teacher perplexity comparison in eval (loads teacher model)")
    ap.add_argument("--skip_judge", action="store_true",
                    help="Skip LLM-as-judge quality eval (eval_quality --judge)")
    # ── Curriculum distillation ───────────────────────────────────────────────────
    ap.add_argument("--curriculum", action="store_true",
                    help="Run SFT warmup (distill_sft.py) before KD stage (pytorch only)")
    ap.add_argument("--sft_epochs", type=int, default=1,
                    help="SFT warmup epochs (default: 1)")
    # ── Synthetic data generation ─────────────────────────────────────────────────
    ap.add_argument("--synthetic_data", action="store_true",
                    help="Generate synthetic data before distillation")
    ap.add_argument("--n_synthetic", type=int, default=2000,
                    help="Number of synthetic pairs to generate (default: 2000)")
    # ── Benchmarks ────────────────────────────────────────────────────────────────
    ap.add_argument("--benchmarks", action="store_true",
                    help="Run WikiText-2 perplexity benchmark after eval")
    ap.add_argument("--baseline_dir", type=str, default=None,
                    help="Previous run dir for regression detection in benchmarks")
    # ── Experiment log ────────────────────────────────────────────────────────────
    ap.add_argument("--log_experiment", action="store_true", default=True,
                    help="Log run to experiment_log.jsonl (default: on)")
    ap.add_argument("--no_log_experiment", dest="log_experiment", action="store_false",
                    help="Disable experiment logging")
    ap.add_argument("--experiment_log", type=str, default="experiment_log.jsonl",
                    help="Path to experiment log file")
    # ── Resource check ────────────────────────────────────────────────────────────
    ap.add_argument("--memory_check", action="store_true",
                    help="Estimate memory usage before starting and warn if near 36GB")
    # ── Multi-trial search ────────────────────────────────────────────────────────
    ap.add_argument("--n_trials", type=int, default=1,
                    help="Run N independent trials; best by eval_perplexity wins (default: 1)")
    # ── Reproducibility ──────────────────────────────────────────────────────────
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed passed to all training scripts (default: 42)")
    # ── Auto naming ──────────────────────────────────────────────────────────────
    ap.add_argument("--auto_name", action="store_true",
                    help="Auto-generate output_dir name from config hash + timestamp")
    args = ap.parse_args()

    setup_logging(args.verbose)
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # Load config overrides
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
                LOG.info("Config override: %s=%s", k, v)

    # Legacy --export-gguf → --export gguf
    if args.export_gguf and args.export == "gguf":
        pass  # already correct default
    elif args.export_gguf:
        args.export = "gguf"

    # Auto-name output_dir from config hash + timestamp
    if args.auto_name:
        import hashlib
        from datetime import datetime as _dt
        cfg_str = (f"{args.backend}-e{args.epochs}-r{args.lora_r}"
                   f"-t{args.temperature}-s{args.max_samples}")
        h = hashlib.md5(cfg_str.encode()).hexdigest()[:6]
        ts = _dt.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = f"./runs/{h}-{ts}"
        LOG.info("Auto-named output_dir: %s", args.output_dir)

    import time as _time
    _start_time = _time.time()
    _steps_completed: list[str] = []

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Backend: %s  Export: %s", args.backend, args.export)

    # ── Step 0: Experiment history ────────────────────────────────────────────────
    # Always import experiment_log (needed for trial-loop propose_next/diagnose)
    import sys as _sys
    _sys.path.insert(0, str(project_root / "scripts"))
    from experiment_log import ExperimentLog, collect_metrics as _collect_metrics_fn

    _exp_log = ExperimentLog(args.experiment_log)
    if args.log_experiment or args.n_trials > 1:
        _summary = _exp_log.summarize(10)
        if "(no experiment history" not in _summary:
            LOG.info("Recent experiment history:\n%s", _summary)
        else:
            LOG.info("No previous experiment history found.")

    # ── Step 0a: Memory check ─────────────────────────────────────────────────────
    if args.memory_check:
        def _estimate_gb(model_id: str, cache_dir) -> float:
            """Rough estimate: 2 bytes/param (bf16) × 4× for training overhead."""
            try:
                import json as _json
                from pathlib import Path as _Path
                from huggingface_hub import cached_assets_path  # noqa: F401
                # Try to read config.json from cache
                import transformers
                cfg_path = None
                for search in [
                    _Path(cache_dir or "") / "models--" / model_id.replace("/", "--"),
                    _Path.home() / ".cache" / "huggingface" / "hub" /
                    ("models--" + model_id.replace("/", "--")),
                ]:
                    for cfg in search.rglob("config.json"):
                        cfg_path = cfg
                        break
                    if cfg_path:
                        break
                if not cfg_path:
                    return 0.0
                with open(cfg_path) as f:
                    cfg = _json.load(f)
                params = cfg.get("num_parameters") or (
                    cfg.get("hidden_size", 0) * cfg.get("num_hidden_layers", 0) * 12
                )
                return params * 2 * 4 / 1e9  # bf16 × 4× training overhead
            except Exception:
                return 0.0

        cache_dir = os.environ.get("HF_HOME") or args.__dict__.get("cache_dir")
        teacher_id = "Qwen/Qwen2-1.5B-Instruct" if args.open else "meta-llama/Llama-3.2-8B-Instruct"
        student_id = "Qwen/Qwen2-0.5B-Instruct" if args.open else "meta-llama/Llama-3.2-1B-Instruct"
        est = _estimate_gb(teacher_id, cache_dir) + _estimate_gb(student_id, cache_dir)
        if est > 0:
            if args.backend == "mlx":
                # MLX frees teacher after logit precompute; peak memory is student
                # + optimizer state + logit tensors (batch × seq_len × vocab_size × 4B).
                # At batch=2, seq=512, vocab=152k: ~0.6 GB per forward pass.
                LOG.info(
                    "Estimated memory (MLX): ~%.1f GB student+optimizer "
                    "(teacher freed after precompute)", _estimate_gb(student_id, cache_dir)
                )
                vocab_gb = args.batch_size * 512 * 152000 * 4 / 1e9
                LOG.info(
                    "Peak logit tensor per micro-batch: ~%.2f GB "
                    "(batch_size=%d × seq_len=512 × vocab=152k × float32)",
                    vocab_gb, args.batch_size,
                )
                if vocab_gb > 2.0:
                    LOG.warning(
                        "Logit tensor %.2f GB > 2 GB — likely OOM cause. "
                        "Reduce --batch_size (currently %d) to 1 or 2.",
                        vocab_gb, args.batch_size,
                    )
            else:
                LOG.info("Estimated memory: ~%.1f GB (teacher + student training)", est)
                if est > 30:
                    LOG.warning(
                        "Memory estimate (%.1f GB) may approach 36GB M3 Max limit. "
                        "Consider --batch_size 2 or --grad_acc 32 if OOM.", est,
                    )

    # ── Step 0b: Dataset filtering ────────────────────────────────────────────────
    _dataset_override: str | None = None
    if args.filter:
        _filter_out = output_dir / "filtered_data"
        LOG.info(
            "Pre-filtering dataset '%s' → top-%d samples ...",
            args.dataset, args.filter_target,
        )
        filter_cmd = [
            sys.executable, "scripts/filter_dataset.py",
            "--dataset", args.dataset,
            "--output_dir", str(_filter_out),
            "--target", str(args.filter_target),
            "--min_response_words", str(args.filter_min_response_words),
            "--min_distinct2", str(args.filter_min_distinct2),
            "--jaccard_threshold", str(args.filter_jaccard),
        ]
        if args.filter_minhash:
            filter_cmd.append("--minhash")
        if args.filter_teacher_score:
            filter_cmd += ["--teacher_score"]
            if args.open:
                filter_cmd += ["--teacher", "Qwen/Qwen2-1.5B-Instruct"]
        if args.offline:
            filter_cmd.append("--offline")
        try:
            run_cmd(filter_cmd, project_root)
            _dataset_override = str(_filter_out)
            _steps_completed.append("filter")
            LOG.info("Filtered dataset ready: %s", _dataset_override)
        except RuntimeError as e:
            LOG.warning("Dataset filtering failed (non-fatal, using raw dataset): %s", e)

    # ── Step 0c: Synthetic data generation ───────────────────────────────────────
    if args.synthetic_data:
        LOG.info("Generating %d synthetic pairs...", args.n_synthetic)
        synth_cmd = [
            sys.executable, "scripts/generate_synthetic_data.py",
            "--output_dir", str(output_dir),
            "--n_generate", str(args.n_synthetic),
        ]
        if args.open:
            synth_cmd.append("--open")
        if args.offline:
            synth_cmd.append("--offline")
        try:
            run_cmd(synth_cmd, project_root)
            _dataset_override = str(output_dir / "synthetic_data")
            _steps_completed.append("synthetic_data")
            LOG.info("Synthetic dataset ready: %s", _dataset_override)
        except RuntimeError as e:
            LOG.warning("Synthetic data generation failed (non-fatal): %s", e)

    # ── Step 0c: SFT curriculum warmup (PyTorch backend only) ────────────────────
    _sft_checkpoint: str | None = None
    if args.curriculum:
        if args.backend != "pytorch":
            LOG.warning(
                "--curriculum is only supported with --backend pytorch. "
                "Skipping SFT warmup for backend '%s'.", args.backend,
            )
        else:
            LOG.info("Running SFT warmup (%d epoch(s))...", args.sft_epochs)
            sft_cmd = [
                sys.executable, "scripts/distill_sft.py",
                "--output_dir", str(output_dir),
                "--epochs", str(args.sft_epochs),
                "--max_samples", str(args.max_samples),
                "--lora_r", str(args.lora_r),
            ]
            if args.open:
                sft_cmd.append("--open")
            if args.offline:
                sft_cmd.append("--offline")
            if args.watchdog:
                sft_cmd.append("--watchdog")
            if _dataset_override:
                sft_cmd += ["--dataset", _dataset_override]
            try:
                run_cmd(sft_cmd, project_root)
                _sft_checkpoint = str(output_dir / "sft_checkpoint")
                _steps_completed.append("sft_warmup")
                LOG.info("SFT checkpoint ready: %s", _sft_checkpoint)
            except RuntimeError as e:
                LOG.warning("SFT warmup failed (non-fatal, continuing without curriculum): %s", e)

    # ── 1. Distillation (single trial or multi-trial loop) ───────────────────────
    if args.n_trials > 1:
        # Multi-trial: run distill+eval N times, export winner only
        LOG.info("Multi-trial mode: %d trials", args.n_trials)
        _trial_results: list[tuple[float, int, Path, dict]] = []
        _base_trial_config = {
            "temperature": args.temperature,
            "lora_r": args.lora_r,
            "epochs": args.epochs,
        }

        import copy as _copy

        for _t in range(args.n_trials):
            # Propose config for this trial
            if _t == 0:
                _tc = dict(_base_trial_config)
            else:
                _tc = _exp_log.propose_next(_base_trial_config)

            # Apply proposed config
            _t_args_temperature = _tc.get("temperature", args.temperature)
            _t_args_lora_r = int(_tc.get("lora_r", args.lora_r))
            _t_args_epochs = int(_tc.get("epochs", args.epochs))
            trial_dir = output_dir / f"trial_{_t:02d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_seed = args.seed + _t

            LOG.info(
                "Trial %d/%d — dir=%s  temp=%.2f  lora_r=%d  epochs=%d  seed=%d",
                _t + 1, args.n_trials, trial_dir,
                _t_args_temperature, _t_args_lora_r, _t_args_epochs, trial_seed,
            )

            _trial_args = _copy.copy(args)
            _trial_args.temperature = _t_args_temperature
            _trial_args.lora_r = _t_args_lora_r
            _trial_args.epochs = _t_args_epochs

            _t_distill_cmd = _build_distill_cmd(
                _trial_args, trial_dir, _sft_checkpoint, _dataset_override, trial_seed
            )
            try:
                run_cmd(_t_distill_cmd, project_root)
            except RuntimeError as e:
                LOG.warning("Trial %d distillation failed: %s — skipping", _t, e)
                _trial_results.append((float("inf"), _t, trial_dir, _tc))
                continue

            # Per-trial perplexity eval (no export, no quality eval yet)
            _t_eval_cmd = [sys.executable, "scripts/run_eval.py", str(trial_dir)]
            if args.open:
                _t_eval_cmd += ["--student", "Qwen/Qwen2-0.5B-Instruct"]
            if args.offline:
                _t_eval_cmd += ["--offline"]
            try:
                run_cmd(_t_eval_cmd, project_root)
            except RuntimeError as e:
                LOG.warning("Trial %d eval failed: %s", _t, e)

            _t_metrics = _collect_metrics_fn(str(trial_dir))
            _t_ppl = _t_metrics.get("eval_perplexity", float("inf"))
            _trial_results.append((_t_ppl, _t, trial_dir, _tc))
            LOG.info("Trial %d result: eval_perplexity=%.3f", _t, _t_ppl)

        # Select winner
        _trial_results.sort(key=lambda x: x[0])
        _best_ppl, _best_idx, _best_dir, _best_config = _trial_results[0]
        LOG.info(
            "Winner: trial_%02d  eval_perplexity=%.3f  config=%s",
            _best_idx, _best_ppl, _best_config,
        )
        # Diagnose winner
        _winner_metrics = _collect_metrics_fn(str(_best_dir))
        _diag = _exp_log.diagnose(_winner_metrics)
        for _d in _diag:
            LOG.info("Diagnosis: %s", _d)

        # Point output_dir at winner for export + logging
        output_dir = _best_dir
        _steps_completed.append("distill")

    else:
        # Single trial (existing behavior)
        distill_cmd = _build_distill_cmd(
            args, output_dir, _sft_checkpoint, _dataset_override, args.seed
        )
        run_cmd(distill_cmd, project_root)
        _steps_completed.append("distill")

    # ── 2. Export ─────────────────────────────────────────────────────────────────
    export_targets = set()
    if args.export == "all":
        export_targets = {"gguf", "coreml", "mlx"}
    elif args.export != "none":
        export_targets = {args.export}

    if "gguf" in export_targets:
        LOG.info("Exporting GGUF...")
        try:
            export_gguf(output_dir, project_root, args.outtype)
        except RuntimeError as e:
            LOG.warning("GGUF export failed (non-fatal): %s", e)

    if "coreml" in export_targets:
        LOG.info("Exporting CoreML...")
        try:
            export_coreml(output_dir, project_root, args.coreml_quantize)
        except RuntimeError as e:
            LOG.warning("CoreML export failed (non-fatal): %s", e)

    if "mlx" in export_targets and args.backend != "mlx":
        # MLX quant from a non-MLX training run (e.g., pytorch → mlx quant)
        LOG.info("Exporting MLX quantization...")
        export_mlx_quant(output_dir, project_root, args.q_bits)

    # ── 3. Perplexity eval (student vs teacher vs quant) ─────────────────────────
    # MLX saves weights as .npz (not HF format) — PyTorch run_eval.py cannot load them
    if not args.skip_eval and args.backend != "mlx":
        LOG.info("Running perplexity eval...")
        eval_cmd = [sys.executable, "scripts/run_eval.py", str(output_dir)]
        if args.open:
            eval_cmd += ["--student", "Qwen/Qwen2-0.5B-Instruct"]
        if args.offline:
            eval_cmd += ["--offline"]
        if args.compare_teacher:
            eval_cmd += ["--compare_teacher"]
            if args.open:
                eval_cmd += ["--teacher", "Qwen/Qwen2-1.5B-Instruct"]
        mlx_quant_dir = output_dir / f"mlx_q{args.q_bits}"
        if mlx_quant_dir.exists():
            eval_cmd += ["--quant_dir", str(mlx_quant_dir)]
            LOG.info("Quant dir found: %s", mlx_quant_dir)
        try:
            run_cmd(eval_cmd, project_root)
        except RuntimeError as e:
            LOG.warning("Perplexity eval failed (non-fatal): %s", e)

    # ── 4. Quality eval (diversity + LLM-as-judge) ───────────────────────────────
    # NOTE: Quality eval only runs on winner (output_dir points to best trial after line 490)
    # This saves 5-10 min per non-winning trial (~20-40 min for 5-trial runs)
    # MLX saves weights as .npz (not HF format) — eval_quality cannot load them
    if not args.skip_eval and not args.skip_judge and args.backend != "mlx":
        LOG.info("Running quality eval (diversity + judge) on winning model...")
        quality_cmd = [sys.executable, "scripts/eval_quality.py", str(output_dir)]
        if args.open:
            quality_cmd += ["--student", "Qwen/Qwen2-0.5B-Instruct"]
        if args.offline:
            quality_cmd += ["--offline"]
        if args.compare_teacher:
            quality_cmd += ["--judge"]
            if args.open:
                quality_cmd += ["--teacher", "Qwen/Qwen2-1.5B-Instruct"]
        try:
            run_cmd(quality_cmd, project_root)
            _steps_completed.append("quality_eval")
        except RuntimeError as e:
            LOG.warning("Quality eval failed (non-fatal): %s", e)

    # ── 5. WikiText-2 benchmark ───────────────────────────────────────────────────
    if args.benchmarks:
        LOG.info("Running WikiText-2 benchmark...")
        bench_cmd = [sys.executable, "scripts/run_benchmarks.py", str(output_dir)]
        if args.open:
            bench_cmd += ["--student", "Qwen/Qwen2-0.5B-Instruct"]
        if args.offline:
            bench_cmd += ["--offline"]
        if args.baseline_dir:
            bench_cmd += ["--baseline_dir", args.baseline_dir]
        try:
            run_cmd(bench_cmd, project_root)
            _steps_completed.append("benchmarks")
        except RuntimeError as e:
            LOG.warning("Benchmark failed (non-fatal): %s", e)

    # ── 5b. Diagnose (single-trial only; multi-trial diagnoses winner above) ──────
    if args.n_trials <= 1 and not args.skip_eval:
        try:
            _final_metrics = _collect_metrics_fn(str(output_dir))
            _diag = _exp_log.diagnose(_final_metrics)
            for _d in _diag:
                LOG.info("Diagnosis: %s", _d)
        except Exception as e:
            LOG.warning("Diagnosis failed (non-fatal): %s", e)

    # ── 6. Log experiment ─────────────────────────────────────────────────────────
    if args.log_experiment:
        try:
            from experiment_log import build_entry, make_run_id
            import torch as _torch

            hardware = (
                "mps" if _torch.backends.mps.is_available() else
                "cuda" if _torch.cuda.is_available() else "cpu"
            )
            teacher_id = ("Qwen/Qwen2-1.5B-Instruct" if args.open
                          else "meta-llama/Llama-3.2-8B-Instruct")
            student_id = ("Qwen/Qwen2-0.5B-Instruct" if args.open
                          else "meta-llama/Llama-3.2-1B-Instruct")
            run_id = make_run_id(str(output_dir))
            config = {
                "backend": args.backend,
                "teacher": teacher_id,
                "student": student_id,
                "dataset": str(_dataset_override or args.dataset),
                "epochs": args.epochs,
                "max_samples": args.max_samples,
                "temperature": args.temperature,
                "lora_r": args.lora_r,
                "curriculum": args.curriculum,
                "sft_epochs": args.sft_epochs if args.curriculum else 0,
                "synthetic_data": args.synthetic_data,
                "n_synthetic": args.n_synthetic if args.synthetic_data else 0,
                "export": args.export,
                "seed": args.seed,
                "n_trials": args.n_trials,
            }
            metrics = _collect_metrics_fn(str(output_dir))
            entry = build_entry(
                run_id=run_id,
                output_dir=str(output_dir),
                config=config,
                metrics=metrics,
                outcome="success",
                steps_completed=_steps_completed,
                start_time=_start_time,
                hardware=hardware,
            )
            _exp_log.append(entry)
            LOG.info("Experiment logged to %s (run_id: %s)", args.experiment_log, run_id)
        except Exception as e:
            LOG.warning("Experiment logging failed (non-fatal): %s", e)

    LOG.info("Agent finished. Output: %s  Steps: %s", output_dir, _steps_completed)


if __name__ == "__main__":
    main()
