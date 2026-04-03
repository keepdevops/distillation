"""Repository and package directory helpers (no sys.path hacks)."""

from __future__ import annotations

from pathlib import Path


def package_dir() -> Path:
    """Directory containing this package (`.../distill`)."""
    return Path(__file__).resolve().parent.parent  # infra/ → distill/


def project_dir() -> Path:
    """Repository root (parent of the `distill` package)."""
    return package_dir().parent


def scripts_dir() -> Path:
    """`scripts/` dir (shell helpers, LaunchAgent assets, etc.)."""
    return project_dir() / "scripts"
