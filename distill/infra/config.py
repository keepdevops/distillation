"""Centralized project configuration with env-var and JSON-file overrides.

Override hierarchy (highest → lowest priority):
  1. Environment variables  (DISTILL_<KEY> or legacy keys like THRESHOLD, HF_HOME)
  2. configs/overrides.json (per-machine config — add to .gitignore)
  3. configs/<section>.json (tracked defaults)
  4. Dataclass field defaults (compile-time fallbacks)

Usage::

    from distill.infra.config import cfg

    root = cfg.paths.llama_cpp_root        # Path | None
    port = cfg.services.gradio_port        # int
    model = cfg.models.open_teacher        # str
    lr = cfg.training.learning_rate        # float
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _env(key: str, default: Any = None) -> Any:
    """Read DISTILL_<KEY> from environment, falling back to *default*."""
    return os.environ.get(f"DISTILL_{key.upper()}", default)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        LOG.error("Failed to parse config file %s: %s", path, exc)
        return {}


def _resolve_llama_root() -> Path | None:
    if env := os.environ.get("LLAMA_CPP_ROOT"):
        p = Path(env)
        if p.exists():
            return p
        LOG.warning("LLAMA_CPP_ROOT=%s does not exist", env)
    candidates = [
        Path("/Users/Shared/llama"),
        Path.home() / "llama.cpp",
        Path.cwd() / "llama.cpp",
        Path.cwd().parent / "llama.cpp",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_mactop() -> str:
    if env := os.environ.get("MACTOP_BIN"):
        return env
    if found := shutil.which("mactop"):
        return found
    brew = Path("/opt/homebrew/bin/mactop")
    if brew.exists():
        return str(brew)
    return "mactop"  # last-resort: rely on PATH at runtime


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    llama_cpp_root: Path | None = field(default_factory=_resolve_llama_root)
    llama_models_dir: Path | None = None
    mactop_bin: str = field(default_factory=_resolve_mactop)
    hf_cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
    )

    def __post_init__(self) -> None:
        if self.llama_models_dir is None and self.llama_cpp_root is not None:
            self.llama_models_dir = self.llama_cpp_root / "models"
        if env := os.environ.get("LLAMA_CPP_MODELS_DIR"):
            self.llama_models_dir = Path(env)


@dataclass
class ServiceConfig:
    gradio_port: int = 7860
    llama_server_port: int = 8089
    llama_teacher_port: int = 8090
    tmux_session: str = "distill"

    def __post_init__(self) -> None:
        if env := _env("port"):
            self.gradio_port = int(env)
        if env := _env("llama_server_port"):
            self.llama_server_port = int(env)
        if env := _env("llama_teacher_port"):
            self.llama_teacher_port = int(env)
        if env := _env("tmux_session"):
            self.tmux_session = env


@dataclass
class ModelConfig:
    open_teacher: str = "Qwen/Qwen2-1.5B-Instruct"
    open_student: str = "Qwen/Qwen2-0.5B-Instruct"
    default_teacher: str = "meta-llama/Llama-3.2-8B-Instruct"
    default_student: str = "meta-llama/Llama-3.2-1B-Instruct"
    default_dataset: str = "tatsu-lab/alpaca"
    cache_models: list = field(default_factory=list)
    cache_datasets: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if env := _env("open_teacher"):
            self.open_teacher = env
        if env := _env("open_student"):
            self.open_student = env
        if env := _env("default_teacher"):
            self.default_teacher = env
        if env := _env("default_student"):
            self.default_student = env


@dataclass
class TrainingConfig:
    batch_size: int = 8
    batch_size_mlx: int = 2
    learning_rate: float = 2e-4
    epochs: int = 2
    max_samples: int = 2000
    grad_accumulation: int = 4
    kd_temperature: float = 1.0
    topk_logits: int = 50
    lora_r: int = 8
    lora_alpha: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    context_size: int = 4096


@dataclass
class EvalConfig:
    n_samples: int = 100
    batch_size: int = 8
    temperature: float = 0.7
    max_length: int = 512
    n_sequences: int = 500


@dataclass
class FilterConfig:
    target_samples: int = 10_000
    min_response_words: int = 30
    min_distinct2: float = 0.40
    max_jaccard: float = 0.55


@dataclass
class ThermalConfig:
    threshold_celsius: float = 85.0
    interval_seconds: int = 30
    fan_curve: list = field(default_factory=lambda: [
        [60, 1200], [70, 2500], [80, 4000], [90, 6000],
    ])

    def __post_init__(self) -> None:
        if env := os.environ.get("THRESHOLD"):
            self.threshold_celsius = float(env)
        if env := os.environ.get("INTERVAL"):
            self.interval_seconds = int(env)


# ── Loader ────────────────────────────────────────────────────────────────────

def _filter_fields(cls: type, raw: dict) -> dict:
    """Keep only keys that are declared fields on *cls*."""
    valid = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


class DistillConfig:
    """Project-wide configuration singleton.

    Loads JSON config files from *config_dir* (default: ``configs/``),
    then applies environment-variable overrides defined in each dataclass.
    Per-machine tuning goes in ``configs/overrides.json`` (git-ignored).
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        d = Path(config_dir) if config_dir else _CONFIG_DIR
        defaults = _load_json(d / "defaults.json")
        models_raw = _load_json(d / "models.json")
        thermal_raw = _load_json(d / "thermal.json")
        overrides = _load_json(d / "overrides.json")

        def _merge(section: str, base: dict | None = None) -> dict:
            merged = dict(base or defaults.get(section, {}))
            merged.update(overrides.get(section, {}))
            return merged

        self.paths = PathConfig()
        self.services = ServiceConfig(**_filter_fields(ServiceConfig, _merge("services")))
        self.models = ModelConfig(**_filter_fields(ModelConfig, {**models_raw, **overrides.get("models", {})}))
        self.training = TrainingConfig(**_filter_fields(TrainingConfig, _merge("training")))
        self.eval = EvalConfig(**_filter_fields(EvalConfig, _merge("eval")))
        self.filter = FilterConfig(**_filter_fields(FilterConfig, _merge("filter")))
        self.thermal = ThermalConfig(**_filter_fields(ThermalConfig, {**thermal_raw, **overrides.get("thermal", {})}))
        self._dir = d

    def reload(self) -> None:
        """Re-read config files (useful in long-running processes)."""
        self.__init__(self._dir)


# ── Singleton ─────────────────────────────────────────────────────────────────

_cfg: DistillConfig | None = None


def get_config(config_dir: Path | None = None) -> DistillConfig:
    """Return the global config singleton (lazy-initialized on first call)."""
    global _cfg
    if _cfg is None:
        _cfg = DistillConfig(config_dir)
    return _cfg


#: Convenience alias — ``from distill.infra.config import cfg``
cfg = get_config()
