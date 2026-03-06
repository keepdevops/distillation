#!/usr/bin/env python3
"""
Artifact detector for distillation outputs.
Detects available formats, metrics, and artifacts to create dynamic UI tabs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

LOG = logging.getLogger(__name__)


def detect_artifacts(output_dir: str) -> Dict[str, Any]:
    """
    Detect all available artifacts and metrics in a distillation output directory.

    Returns a dict with:
    - formats: List of available model formats (pytorch, mlx, gguf, coreml)
    - has_metrics: Whether training metrics are available
    - has_trainer_state: Whether trainer_state.json exists
    - metrics_file: Path to metrics.jsonl if it exists
    - training_method: Detected training method (minillm, sft, forward, mlx, unsloth)
    - artifacts: List of (name, type, path, size_gb) for each artifact
    - metric_types: Available metric types (loss, eval_loss, reward, entropy, etc.)
    """
    path = Path(output_dir)
    if not path.exists():
        return {"error": f"Directory not found: {output_dir}"}

    result = {
        "formats": [],
        "has_metrics": False,
        "has_trainer_state": False,
        "metrics_file": None,
        "training_method": "unknown",
        "artifacts": [],
        "metric_types": set(),
        "checkpoints": []
    }

    # Detect model formats
    # PyTorch/HuggingFace
    has_config = (path / "config.json").exists()
    has_pytorch_weights = (
        list(path.glob("*.safetensors")) or
        list(path.glob("model*.bin")) or
        (path / "pytorch_model.bin").exists()
    )
    has_adapter = (path / "adapter_model.bin").exists() or (path / "adapter_config.json").exists()

    if has_config and (has_pytorch_weights or has_adapter):
        result["formats"].append("pytorch")

    # MLX
    mlx_weights = list(path.glob("*.npz"))
    if mlx_weights:
        result["formats"].append("mlx")
        for w in mlx_weights:
            size_gb = w.stat().st_size / (1024 ** 3)
            result["artifacts"].append((w.name, "mlx", str(w), size_gb))

    # Check for MLX quantized subdirectories
    for subdir in path.iterdir():
        if subdir.is_dir() and list(subdir.glob("*.npz")) and (subdir / "config.json").exists():
            result["formats"].append("mlx_quant")
            size_gb = sum(f.stat().st_size for f in subdir.glob("*.npz")) / (1024 ** 3)
            result["artifacts"].append((f"{subdir.name}/", "mlx_quant", str(subdir), size_gb))

    # GGUF
    gguf_files = list(path.glob("*.gguf"))
    if gguf_files:
        result["formats"].append("gguf")
        for g in gguf_files:
            size_gb = g.stat().st_size / (1024 ** 3)
            result["artifacts"].append((g.name, "gguf", str(g), size_gb))

    # CoreML
    mlpackages = list(path.glob("*.mlpackage"))
    if mlpackages:
        result["formats"].append("coreml")
        for pkg in mlpackages:
            try:
                size_gb = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / (1024 ** 3)
            except OSError:
                size_gb = 0.0
            result["artifacts"].append((pkg.name, "coreml", str(pkg), size_gb))

    # Detect metrics
    trainer_state_path = path / "trainer_state.json"
    if trainer_state_path.exists():
        result["has_trainer_state"] = True
        result["has_metrics"] = True

    metrics_jsonl_path = path / "metrics.jsonl"
    if metrics_jsonl_path.exists():
        result["has_metrics"] = True
        result["metrics_file"] = str(metrics_jsonl_path)

        # Parse first few lines to detect metric types
        try:
            with open(metrics_jsonl_path) as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Sample first 10 lines
                        break
                    try:
                        entry = json.loads(line.strip())
                        result["metric_types"].update(entry.keys())
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            LOG.warning(f"Could not read metrics.jsonl: {e}")

    # Convert set to sorted list for JSON serialization
    result["metric_types"] = sorted(result["metric_types"])

    # Detect training method from metrics
    metric_types = set(result["metric_types"])
    if "eval_rewards/dummy_reward_func/mean" in metric_types or "eval_entropy" in metric_types:
        result["training_method"] = "minillm"
    elif "eval_kl_div" in metric_types:
        result["training_method"] = "forward"
    elif any("mlx" in fmt for fmt in result["formats"]) and not has_pytorch_weights:
        result["training_method"] = "mlx"
    elif has_adapter:
        # Could be SFT, MiniLLM, or Unsloth (all use LoRA)
        if "unsloth" in str(path).lower():
            result["training_method"] = "unsloth"
        elif result["training_method"] == "minillm":
            pass  # Already detected
        else:
            result["training_method"] = "sft"

    # Detect checkpoints
    for item in path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                result["checkpoints"].append({"path": str(item), "step": step})
            except (IndexError, ValueError):
                pass

    result["checkpoints"].sort(key=lambda x: x["step"])

    return result


def load_all_metrics(output_dir: str) -> List[Dict[str, Any]]:
    """
    Load all metrics from trainer_state.json and metrics.jsonl.
    Merges them by step number.

    Returns:
        List of metric dicts sorted by step
    """
    path = Path(output_dir)
    rows_by_step = {}

    # Load from trainer_state.json
    trainer_state_path = path / "trainer_state.json"
    if trainer_state_path.exists():
        try:
            with open(trainer_state_path) as f:
                state = json.load(f)
            for entry in state.get("log_history", []):
                step = entry.get("step")
                if step is not None:
                    rows_by_step.setdefault(step, {}).update(entry)
        except (json.JSONDecodeError, OSError) as e:
            LOG.warning(f"Could not read trainer_state.json: {e}")

    # Load from metrics.jsonl
    metrics_jsonl_path = path / "metrics.jsonl"
    if metrics_jsonl_path.exists():
        try:
            with open(metrics_jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    step = entry.get("step")
                    if step is not None:
                        rows_by_step.setdefault(step, {}).update(entry)
        except OSError as e:
            LOG.warning(f"Could not read metrics.jsonl: {e}")

    return [rows_by_step[s] for s in sorted(rows_by_step)]


def format_artifact_summary(artifacts: List[tuple]) -> str:
    """Format artifact list as human-readable summary."""
    if not artifacts:
        return "No export artifacts found"

    lines = ["Available artifacts:"]
    by_type = {}
    for name, typ, path, size_gb in artifacts:
        by_type.setdefault(typ, []).append((name, size_gb))

    for typ in sorted(by_type.keys()):
        items = by_type[typ]
        lines.append(f"  {typ.upper()}:")
        for name, size_gb in items:
            lines.append(f"    - {name} ({size_gb:.2f} GB)")

    return "\n".join(lines)
