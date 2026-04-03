"""MLX model path resolution and availability check utilities."""
from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)

# Pre-converted local MLX model paths (avoids re-conversion on each run)
_MLX_LOCAL_PATHS: dict[str, str] = {
    "Qwen/Qwen2-1.5B-Instruct": "airgap_bundle/mlx_models/qwen2-1.5b-instruct",
    "Qwen/Qwen2-0.5B-Instruct": "airgap_bundle/mlx_models/qwen2-0.5b-instruct",
    "meta-llama/Llama-3.2-8B-Instruct": "airgap_bundle/mlx_models/llama-3.2-8b-instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "airgap_bundle/mlx_models/llama-3.2-1b-instruct",
}


def resolve_mlx_path(model_id: str) -> str:
    """Return local MLX path if pre-converted, otherwise the original HF model ID."""
    rel = _MLX_LOCAL_PATHS.get(model_id)
    if rel:
        for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
            candidate = base / rel
            if candidate.exists() and any(candidate.iterdir()):
                LOG.info("Using local MLX model: %s", candidate)
                return str(candidate)
    return model_id


def check_mlx() -> bool:
    """Return True if mlx and mlx-lm are importable."""
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        LOG.error(
            "mlx and mlx-lm are required. Install with:\n"
            "  pip install mlx mlx-lm\n"
            "Note: MLX is Apple-only (M1/M2/M3 Silicon)."
        )
        return False
