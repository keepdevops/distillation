#!/usr/bin/env python3
"""
Export a HuggingFace model to CoreML (.mlpackage) for Apple Neural Engine inference.

Pipeline:
  1. Load model + tokenizer via HuggingFace
  2. TorchScript trace with example inputs
  3. coremltools.convert → .mlpackage (ComputeUnit.ALL = CPU + GPU + ANE)
  4. Optional quantization: int4 / int8 / float16

Usage:
  python scripts/export_coreml.py --model_dir ./distilled-minillm
  python scripts/export_coreml.py --model_dir ./distilled-minillm --quantize int4
  python scripts/export_coreml.py --model_dir Qwen/Qwen2-0.5B-Instruct --output_dir ./coreml_out

Requirements:
  pip install coremltools>=8.0
"""

import argparse
import logging
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SWIFT_SNIPPET_TEMPLATE = """
// ── Swift inference snippet ────────────────────────────────────────────────
// Add {package_name} to your Xcode project, then:

import CoreML
import NaturalLanguage

let config = MLModelConfiguration()
config.computeUnits = .all  // CPU + GPU + Apple Neural Engine

let model = try {class_name}(configuration: config)

// Prepare input IDs (tokenize externally or use NLTokenizer)
// let inputIDs = MLMultiArray(...)  // shape: [1, seqLen], dtype: Int32

// let input = {class_name}Input(input_ids: inputIDs)
// let output = try model.prediction(input: input)
// let logits = output.logits  // shape: [1, seqLen, vocabSize]
// ──────────────────────────────────────────────────────────────────────────
"""


def parse_args():
    p = argparse.ArgumentParser(description="Export HF model to CoreML .mlpackage")
    p.add_argument("--model_dir", type=str, required=True,
                   help="HF model directory or hub model ID")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: same as model_dir parent)")
    p.add_argument("--quantize", type=str, default=None,
                   choices=["int4", "int8", "float16"],
                   help="Post-training quantization type")
    p.add_argument("--seq_len", type=int, default=128,
                   help="Sequence length for tracing (shorter = faster, longer = more flexible)")
    p.add_argument("--compute_units", type=str, default="ALL",
                   choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
                   help="CoreML compute units (ALL = CPU+GPU+ANE)")
    p.add_argument("--offline", action="store_true", help="Air-gapped: local cache only")
    return p.parse_args()


def _check_coremltools():
    try:
        import coremltools as ct  # noqa: F401
        return True
    except ImportError:
        LOG.error(
            "coremltools is not installed.\n"
            "Install with: pip install 'coremltools>=8.0'\n"
            "Note: coremltools works on macOS only."
        )
        return False


def _check_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        LOG.error("torch is required. Install with: pip install torch")
        return False


def load_model_and_tokenizer(model_dir: str, offline: bool):
    """Load HF model and tokenizer for tracing."""
    import os
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    LOG.info("Loading tokenizer: %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    LOG.info("Loading model: %s", model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,  # float32 required for tracing
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


class _TracingWrapper(object):
    """
    Thin wrapper so TorchScript trace only sees (input_ids,) → logits.
    CoreML expects a simple signature.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, input_ids):
        out = self.model(input_ids=input_ids)
        return out.logits


def trace_model(model, seq_len: int):
    """Trace the model to TorchScript."""
    import torch

    wrapper = _TracingWrapper(model)

    # Example input: batch=1, seq_len tokens
    example_input = torch.zeros((1, seq_len), dtype=torch.int32)

    LOG.info("Tracing model (seq_len=%d)...", seq_len)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_input, strict=False)

    LOG.info("Trace complete.")
    return traced, example_input


def convert_to_coreml(traced, example_input, compute_units_str: str):
    """Convert TorchScript to CoreML model."""
    import coremltools as ct

    compute_units_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    compute_units = compute_units_map.get(compute_units_str, ct.ComputeUnit.ALL)

    LOG.info("Converting to CoreML (compute_units=%s)...", compute_units_str)

    input_shape = ct.Shape(shape=(1, ct.RangeDim(1, 512)))

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=input_shape, dtype=int)],
        outputs=[ct.TensorType(name="logits")],
        compute_units=compute_units,
        minimum_deployment_target=ct.target.iOS17,
    )
    LOG.info("CoreML conversion complete.")
    return mlmodel


def apply_quantization(mlmodel, quantize: str):
    """Apply post-training quantization to CoreML model."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpLinearQuantizerConfig,
        OpPalettizerConfig,
        linear_quantize_weights,
        palettize_weights,
    )

    LOG.info("Applying quantization: %s", quantize)

    if quantize == "float16":
        mlmodel = ct.models.MLModel(
            mlmodel.get_spec(),
            compute_units=mlmodel.compute_unit,
        )
        # Float16 via compression
        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="float16")
        config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config=config)

    elif quantize == "int8":
        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config=config)

    elif quantize == "int4":
        # 4-bit palettization (lookup table quantization)
        op_config = OpPalettizerConfig(mode="kmeans", nbits=4)
        config = OptimizationConfig(global_config=op_config)
        mlmodel = palettize_weights(mlmodel, config=config)

    LOG.info("Quantization applied.")
    return mlmodel


def main():
    args = parse_args()

    if not _check_coremltools():
        sys.exit(1)
    if not _check_torch():
        sys.exit(1)

    model_path = Path(args.model_dir)
    model_name = model_path.name if model_path.exists() else args.model_dir.split("/")[-1]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif model_path.exists():
        output_dir = model_path.parent
    else:
        output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_suffix = f"_{args.quantize}" if args.quantize else ""
    package_name = f"{model_name}{quant_suffix}.mlpackage"
    output_path = output_dir / package_name

    # Load
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.offline)

    # Trace
    traced, example_input = trace_model(model, args.seq_len)

    # Convert
    mlmodel = convert_to_coreml(traced, example_input, args.compute_units)

    # Quantize
    if args.quantize:
        mlmodel = apply_quantization(mlmodel, args.quantize)

    # Save
    LOG.info("Saving .mlpackage: %s", output_path)
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 ** 2)
    LOG.info("Saved %s (%.1f MB)", output_path, size_mb)

    # Swift snippet
    class_name = model_name.replace("-", "_").replace(".", "_")
    snippet = SWIFT_SNIPPET_TEMPLATE.format(
        package_name=package_name,
        class_name=class_name,
    )
    print(snippet)

    LOG.info("CoreML export complete: %s", output_path)


if __name__ == "__main__":
    main()
