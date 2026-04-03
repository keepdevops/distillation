"""Model loading and discovery helpers for the distillation dashboard."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from ..backends.universal_loader import UniversalModelLoader

def select_and_load_model(path, model_state):
    """Load model into model_state using universal loader; return status string.

    path can be a local directory, GGUF file, or a HuggingFace model ID.
    model_state is [loader, backend] where loader is UniversalModelLoader instance.
    """
    import logging
    log = logging.getLogger(__name__)
    if not path or "(no " in path:
        return "Select a model above."

    path = path.strip()

    # Determine if it's a local path or HF id
    is_local = os.path.exists(path)
    if is_local:
        path = os.path.abspath(path)
        label = Path(path).name
    else:
        # Treat as HuggingFace model id - use PyTorch backend
        label = path

    try:
        # Create or reuse loader
        if model_state[0] is None or not isinstance(model_state[0], UniversalModelLoader):
            loader = UniversalModelLoader()
            model_state[0] = loader
        else:
            loader = model_state[0]

        # Auto-detect format for local paths, use PyTorch for HF ids
        backend = None if is_local else "pytorch"

        success, message = loader.load(path, backend=backend)

        if success:
            info = loader.get_info()
            model_state[1] = info['backend']  # Store backend type
            log.info("Loaded model: %s (backend: %s)", label, info['backend'])
            status = f"Loaded: {label}\nBackend: {info['backend']}"
            if 'device' in info:
                status += f"\nDevice: {info['device']}"
            return status
        else:
            log.warning("Model load failed %s: %s", path, message)
            return f"Failed to load '{label}': {message}"

    except Exception as e:
        log.warning("Model load failed %s: %s", path, e)
        return f"Failed to load '{label}': {e}"


def _diversity_metrics(text):
    """Return (distinct_1, distinct_2, max_rep) for a generated text."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0, 0.0, 0
    d1 = len(set(tokens)) / len(tokens)
    bigrams = list(zip(tokens, tokens[1:]))
    d2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    max_run = run = 1
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        max_run = max(max_run, run)
    return d1, d2, max_run if len(tokens) > 1 else 0


def _is_hf_model_dir(d: Path) -> bool:
    """Return True if d looks like a complete HuggingFace model directory."""
    if not (d / "config.json").exists():
        return False
    return bool(
        list(d.glob("*.safetensors"))
        or list(d.glob("model*.bin"))
        or (d / "pytorch_model.bin").exists()
    )


def _scan_hf_hub_cache(hub_root: Path) -> list[tuple[str, str]]:
    """Yield (display_label, abs_path) for every model snapshot in an HF hub cache dir."""
    results = []
    if not hub_root.exists():
        return results
    try:
        for entry in hub_root.iterdir():
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            # models--Org--Name  →  Org/Name
            label = entry.name[len("models--"):].replace("--", "/", 1)
            snaps = entry / "snapshots"
            if not snaps.exists():
                continue
            try:
                for snap in sorted(snaps.iterdir(),
                                   key=lambda p: p.stat().st_mtime, reverse=True):
                    if snap.is_dir() and _is_hf_model_dir(snap):
                        results.append((label, str(snap)))
                        break  # only latest snapshot per model
            except PermissionError:
                continue
    except PermissionError:
        pass
    return results


def _discover_all_models(runs_dir: str) -> list[tuple[str, str]]:
    """Return (display_label, abs_path) for every usable model, deduplicated.

    Sources (in priority order — earlier sources win on label conflicts):
      1. Local trained outputs under runs_dir (up to 3 levels deep)
      2. System HF hub cache  (~/.cache/huggingface/hub  or $HF_HOME/hub)
      3. Project-local hf_cache/  next to runs_dir
    """
    seen_paths: set[str] = set()
    seen_labels: set[str] = set()
    results: list[tuple[str, str]] = []

    def _add(label: str, path: str) -> None:
        """Add (label, path) deduplicating by both path and label."""
        if path in seen_paths or not Path(path).exists():
            return
        # For HF cache models: deduplicate by label (same model, different cache copy)
        if label in seen_labels:
            return
        seen_paths.add(path)
        seen_labels.add(label)
        results.append((label, path))

    # ── 1. Local trained outputs ─────────────────────────────────────────────
    root = Path(runs_dir).resolve()
    candidates: list[Path] = []
    try:
        candidates = [root] + [
            d for d in root.rglob("*")
            if d.is_dir() and len(d.relative_to(root).parts) <= 3
        ]
    except PermissionError:
        pass
    # Sort newest-first so most recent trained models appear first
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for d in candidates:
        if _is_hf_model_dir(d):
            try:
                rel = d.relative_to(root)
                label = str(rel) if str(rel) != "." else d.name
            except ValueError:
                label = d.name
            _add(label, str(d))

    # ── 2. System HF hub cache ────────────────────────────────────────────────
    hf_home = os.environ.get("HF_HOME") or str(
        Path.home() / ".cache" / "huggingface"
    )
    for label, path in _scan_hf_hub_cache(Path(hf_home) / "hub"):
        _add(label, path)

    # ── 3. Project-local hf_cache dirs ───────────────────────────────────────
    for local_cache in [
        root / "hf_cache",
        root.parent / "hf_cache",
        root / "scripts" / "hf_cache",
    ]:
        for label, path in _scan_hf_hub_cache(local_cache / "hub"):
            _add(label, path)
        for label, path in _scan_hf_hub_cache(local_cache):
            _add(label, path)

    return results


