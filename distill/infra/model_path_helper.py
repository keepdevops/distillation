#!/usr/bin/env python3
"""
Helper for resolving model paths with MODEL_PATH environment variable support.
"""

import os
from pathlib import Path
from typing import Optional


def get_model_base_path() -> Path:
    """
    Get the base path for model storage.

    Returns MODEL_PATH env var if set, otherwise current directory.
    """
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        return Path(model_path)
    return Path.cwd()


def resolve_model_path(relative_path: str) -> str:
    """
    Resolve a model path relative to MODEL_PATH if set.

    Args:
        relative_path: Path like "distilled-minillm" or "/absolute/path"

    Returns:
        Absolute path string
    """
    path = Path(relative_path)

    # If already absolute, return as-is
    if path.is_absolute():
        return str(path)

    # Try MODEL_PATH first
    model_base = get_model_base_path()
    if model_base != Path.cwd():
        candidate = model_base / relative_path
        if candidate.exists():
            return str(candidate.resolve())

    # Fall back to current directory
    candidate = Path.cwd() / relative_path
    if candidate.exists():
        return str(candidate.resolve())

    # If nothing exists, prefer MODEL_PATH location
    if model_base != Path.cwd():
        return str((model_base / relative_path).resolve())

    return str(candidate.resolve())


def list_available_models() -> list[tuple[str, str]]:
    """
    List all available models in MODEL_PATH and current directory.

    Returns:
        List of (name, path) tuples
    """
    models = []
    seen_paths = set()

    # Check MODEL_PATH
    model_base = get_model_base_path()
    if model_base.exists():
        for item in model_base.iterdir():
            if item.is_dir():
                # Check if it looks like a model dir
                if (item / "config.json").exists() or \
                   list(item.glob("*.gguf")) or \
                   list(item.glob("*.npz")) or \
                   (item / "metrics.jsonl").exists():
                    abs_path = str(item.resolve())
                    if abs_path not in seen_paths:
                        models.append((item.name, abs_path))
                        seen_paths.add(abs_path)

    # Check current directory
    cwd = Path.cwd()
    if cwd != model_base:
        for item in cwd.iterdir():
            if item.is_dir() and item.name.startswith("distilled-"):
                abs_path = str(item.resolve())
                if abs_path not in seen_paths:
                    models.append((item.name, abs_path))
                    seen_paths.add(abs_path)

    return sorted(models)


def get_hf_cache_path() -> Optional[Path]:
    """
    Get the HuggingFace cache directory.

    Checks in order:
    1. HF_HOME env var
    2. MODEL_PATH/hf_cache
    3. ./scripts/hf_cache
    4. ~/.cache/huggingface
    """
    # HF_HOME takes precedence
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        path = Path(hf_home)
        if path.exists():
            return path

    # Check MODEL_PATH/hf_cache
    model_base = get_model_base_path()
    hf_cache = model_base / "hf_cache"
    if hf_cache.exists():
        return hf_cache

    # Check local scripts/hf_cache
    local_cache = Path("scripts/hf_cache")
    if local_cache.exists():
        return local_cache

    # Default HF location
    default_hf = Path.home() / ".cache" / "huggingface"
    if default_hf.exists():
        return default_hf

    return None


if __name__ == "__main__":
    print("Model Path Helper")
    print("=" * 60)
    print(f"MODEL_PATH env: {os.environ.get('MODEL_PATH', '(not set)')}")
    print(f"Base path: {get_model_base_path()}")
    print(f"HF cache: {get_hf_cache_path()}")
    print()
    print("Available models:")
    for name, path in list_available_models():
        print(f"  {name}: {path}")
