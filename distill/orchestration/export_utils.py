"""Model export utilities: GGUF, CoreML, and MLX quantization."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from .subprocess_runner import run_cmd, find_llama_cpp

LOG = logging.getLogger(__name__)


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

    Merges LoRA adapters into base weights, then saves as safetensors.
    LoRA naming from mlx_lm LoRALinear:
      base weight : <layer>.linear.weight
      LoRA A      : <layer>.lora_a.weight  shape (r, in_features)
      LoRA B      : <layer>.lora_b.weight  shape (out_features, r)
      merge       : W + scale * (B @ A)
    where scale = alpha / r = 20.0 / lora_r (standard LoRA, mlx_lm convention).
    """
    import numpy as np

    with open(output_dir / "distill_config.json") as f:
        dcfg = json.load(f)
    student_id = dcfg["student"]
    lora_r = int(dcfg.get("lora_r", 16))
    lora_scale = 20.0 / lora_r  # alpha=20 (hardcoded in mlx backend), scale = alpha/r

    npz_path = output_dir / "mlx_student_weights.npz"
    LOG.info("Loading MLX weights from %s", npz_path)
    try:
        import mlx.core as mx
        raw = mx.load(str(npz_path))
        weights = {k: np.array(v.astype(mx.float32)) for k, v in raw.items()}
    except Exception:
        weights = {k: v.astype(np.float32) for k, v in np.load(str(npz_path)).items()}

    all_keys = set(weights)
    hf_weights: dict[str, "np.ndarray"] = {}
    processed: set[str] = set()

    for key in sorted(all_keys):
        if key in processed:
            continue
        if key.endswith(".lora_a") or key.endswith(".lora_b"):
            processed.add(key)
            continue
        if key.endswith(".linear.weight"):
            prefix = key[: -len(".linear.weight")]
            a_key = f"{prefix}.lora_a"
            b_key = f"{prefix}.lora_b"
            hf_key = f"{prefix}.weight"
            W = weights[key]
            if a_key in all_keys and b_key in all_keys:
                A = weights[a_key]
                B = weights[b_key]
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

    If output_dir contains MLX weights (.npz), merges LoRA and creates a
    temporary HF-compatible directory first.
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
    gguf_dir = Path("/Users/Shared/llama/models")
    gguf_dir.mkdir(parents=True, exist_ok=True)
    out_file = gguf_dir / out_name
    run_cmd(
        [sys.executable, str(convert_script), str(convert_dir),
         "--outfile", str(out_file), "--outtype", outtype],
        project_root,
    )
    LOG.info("GGUF saved: %s", out_file)

    if outtype in ("f16", "bf16", "f32"):
        quantize_bin = Path("/Users/Shared/llama/llama-quantize")
        if not quantize_bin.exists():
            quantize_bin = llama_cpp / "llama-quantize"
        if quantize_bin.exists():
            q4_file = gguf_dir / (output_dir.name + "-Q4_K_M.gguf")
            LOG.info("Running llama-quantize → Q4_K_M: %s", q4_file)
            run_cmd([str(quantize_bin), str(out_file), str(q4_file), "Q4_K_M"], project_root)
            LOG.info("Q4_K_M saved: %s (deploy this one)", q4_file)
            LOG.info("Serve with: /Users/Shared/llama/llama-server -m %s", q4_file)
        else:
            LOG.warning(
                "llama-quantize not found at %s — skipping Q4_K_M quantization.",
                quantize_bin,
            )
            LOG.info("Serve with: /Users/Shared/llama/llama-server -m %s", out_file)
    else:
        LOG.info("Serve with: /Users/Shared/llama/llama-server -m %s", out_file)


def export_coreml(output_dir: Path, project_root: Path, quantize: str | None) -> None:
    """Export model to CoreML .mlpackage."""
    cmd = [
        sys.executable, "-m", "distill.export.coreml",
        "--model_dir", str(output_dir),
        "--output_dir", str(output_dir),
        "--compute_units", "CPU_AND_NE",
    ]
    if quantize and quantize in ("int4", "int8", "float16"):
        cmd += ["--quantize", quantize]
    run_cmd(cmd, project_root)


def export_mlx_quant(output_dir: Path, project_root: Path, q_bits: int) -> None:
    """Quantize a HF-format model directory via mlx_lm.convert."""
    try:
        import shutil
        from mlx_lm import convert as mlx_convert
        quant_dir = output_dir / f"mlx_q{q_bits}"
        if quant_dir.exists():
            LOG.info("Removing existing quantized dir: %s", quant_dir)
            shutil.rmtree(quant_dir)
        LOG.info("MLX quantization → %s", quant_dir)
        mlx_convert(str(output_dir), quantize=True, q_bits=q_bits, q_group_size=64, mlx_path=str(quant_dir))
        LOG.info("MLX quantized model saved: %s", quant_dir)
    except ImportError:
        LOG.warning("mlx_lm not installed; skipping MLX quantization export")
