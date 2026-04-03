"""Discover teachers, students, and datasets from local caches."""
from __future__ import annotations

import os
from pathlib import Path

from .presets import KNOWN_DATASETS, KNOWN_STUDENTS, KNOWN_TEACHERS
from ..infra.paths import project_dir, scripts_dir

PROJECT_DIR = project_dir()
SCRIPTS_DIR = scripts_dir()

def _is_hf_model_dir(d: Path) -> bool:
    if not (d / "config.json").exists():
        return False
    return bool(
        list(d.glob("*.safetensors"))
        or list(d.glob("model*.bin"))
        or (d / "pytorch_model.bin").exists()
    )


def _scan_hf_hub_cache(hub_root: Path) -> list[str]:
    """Return HF model IDs found in a hub cache directory."""
    results = []
    if not hub_root.exists():
        return results
    try:
        for entry in sorted(hub_root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            label = entry.name[len("models--"):].replace("--", "/", 1)
            snaps = entry / "snapshots"
            if not snaps.exists():
                continue
            try:
                for snap in sorted(snaps.iterdir(),
                                   key=lambda p: p.stat().st_mtime, reverse=True):
                    if snap.is_dir() and _is_hf_model_dir(snap):
                        results.append(label)
                        break
            except PermissionError:
                continue
    except PermissionError:
        pass
    return results


def _scan_datasets_cache(cache_root: Path) -> list[str]:
    """Return dataset IDs found in an HF datasets cache directory."""
    results = []
    if not cache_root.exists():
        return results
    try:
        for entry in sorted(cache_root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("datasets--"):
                continue
            label = entry.name[len("datasets--"):].replace("--", "/", 1)
            results.append(label)
    except PermissionError:
        pass
    return results


def _scan_local_checkpoints(search_root: Path, _max_depth: int = 5) -> list[str]:
    """Return local directories that look like HF model checkpoints.

    Bounded to _max_depth levels below search_root to avoid unbounded rglob
    over the entire filesystem when search_root is the project directory.
    """
    results = []
    if not search_root.exists():
        return results
    try:
        candidates = sorted(
            (
                p for p in search_root.rglob("config.json")
                if len(p.relative_to(search_root).parts) <= _max_depth
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in candidates:
            parent = d.parent
            if _is_hf_model_dir(parent):
                results.append(str(parent))
    except (PermissionError, OSError):
        pass
    return results


def discover_teachers() -> list[str]:
    seen, result = set(), []
    def add(v):
        if v not in seen:
            seen.add(v); result.append(v)

    # Local checkpoints (trained outputs)
    for p in _scan_local_checkpoints(PROJECT_DIR):
        add(p)

    # HF hub caches
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    for m in _scan_hf_hub_cache(Path(hf_home) / "hub"):
        add(m)
    for m in _scan_hf_hub_cache(SCRIPTS_DIR / "hf_cache"):
        add(m)

    # Known presets (fill gaps)
    for m in KNOWN_TEACHERS:
        add(m)
    return result


def discover_students() -> list[str]:
    seen, result = set(), []
    def add(v):
        if v not in seen:
            seen.add(v); result.append(v)

    # SFT checkpoint first
    sft = PROJECT_DIR / "distilled-minillm" / "sft_checkpoint"
    if _is_hf_model_dir(sft):
        add(str(sft))

    # All local checkpoints
    for p in _scan_local_checkpoints(PROJECT_DIR):
        add(p)

    # HF hub caches
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    for m in _scan_hf_hub_cache(Path(hf_home) / "hub"):
        add(m)
    for m in _scan_hf_hub_cache(SCRIPTS_DIR / "hf_cache"):
        add(m)

    # Known presets
    for m in KNOWN_STUDENTS:
        add(m)
    return result


def discover_datasets() -> list[str]:
    seen, result = set(), []
    def add(v):
        if v not in seen:
            seen.add(v); result.append(v)

    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    for ds in _scan_datasets_cache(Path(hf_home) / "datasets"):
        add(ds)
    for ds in _scan_datasets_cache(SCRIPTS_DIR / "datasets_cache"):
        add(ds)

    for ds in KNOWN_DATASETS:
        add(ds)
    return result


def discover_output_dirs() -> list[str]:
    result = []
    try:
        for d in sorted(PROJECT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir() and not d.name.startswith(".") and d.name != "scripts":
                result.append(str(d))
    except OSError:
        pass
    default = str(PROJECT_DIR / "distilled-minillm")
    if default not in result:
        result.insert(0, default)
    return result
