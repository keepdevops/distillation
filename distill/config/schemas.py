"""Pydantic schemas mirroring all C++ structs — single source of truth.

These models validate configs coming from the UI (YAML editor, preset loader,
API calls) before they reach the training backends or export engine. They mirror
the C++ struct fields so the two layers stay in sync.

Import pattern:
    from distill.config.schemas import LoRAConfig, TrainingConfig, ExportConfig
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── LoRA / QLoRA ──────────────────────────────────────────────────────────────

class LoRAConfig(BaseModel):
    """Mirrors distill_cpp.LoRAConfig."""

    rank: int        = Field(16,    ge=1,   le=512,  description="LoRA rank r")
    alpha: int       = Field(32,    ge=1,   le=1024, description="LoRA scaling alpha")
    dropout: float   = Field(0.05,  ge=0.0, le=0.5,  description="LoRA dropout rate")
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="Which linear layers to adapt",
    )
    use_qlora: bool  = Field(False, description="Use 4-bit QLoRA base model loading")
    qlora_bits: int  = Field(4, ge=4, le=8, description="Bit width for QLoRA base")
    bias: str        = Field("none", pattern=r"^(none|all|lora_only)$")
    task_type: str   = Field("CAUSAL_LM")

    @model_validator(mode="after")
    def alpha_gte_rank(self) -> "LoRAConfig":
        if self.alpha < self.rank:
            self.alpha = self.rank * 2
        return self

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def estimated_params(self, hidden_size: int = 2048, num_layers: int = 24) -> int:
        """Rough estimate of trainable LoRA parameter count."""
        params_per_layer = 2 * hidden_size * self.rank * len(self.target_modules) // 2
        return params_per_layer * num_layers

    def estimated_vram_mb(self, hidden_size: int = 2048, num_layers: int = 24,
                           dtype_bytes: int = 2) -> float:
        """Estimated VRAM for LoRA adapters in MB."""
        params = self.estimated_params(hidden_size, num_layers)
        return (params * dtype_bytes) / (1024 ** 2)


# ── Training ──────────────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    """Full training run configuration."""

    backend: Literal["mlx", "sft", "minillm", "unsloth", "forward"] = "mlx"
    teacher: str    = Field(..., min_length=3)
    student: str    = Field(..., min_length=3)
    dataset: str    = Field("yahma/alpaca-cleaned", min_length=3)
    output_dir: str = Field("outputs/distilled")
    epochs: int     = Field(3, ge=1, le=200)
    lr: float       = Field(2e-4, gt=0.0, lt=1.0)
    batch_size: int = Field(4, ge=1, le=256)
    grad_accum: int = Field(4, ge=1, le=128)
    max_length: int = Field(512, ge=64, le=32768)
    warmup_steps: int = Field(100, ge=0)
    lora: LoRAConfig  = Field(default_factory=LoRAConfig)
    watchdog: bool  = Field(False)
    offline: bool   = Field(False)
    export: str     = Field("gguf", pattern=r"^(none|gguf|mlx|coreml|all)$")

    @field_validator("lr", mode="before")
    @classmethod
    def parse_lr(cls, v: Any) -> float:
        return float(v)

    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    def to_cli_args(self) -> list[str]:
        """Return agent CLI argument list."""
        args = [
            "--backend",    self.backend,
            "--teacher",    self.teacher,
            "--student",    self.student,
            "--dataset",    self.dataset,
            "--output_dir", self.output_dir,
            "--epochs",     str(self.epochs),
            "--lr",         str(self.lr),
            "--batch_size", str(self.batch_size),
            "--lora_r",     str(self.lora.rank),
            "--lora_alpha", str(self.lora.alpha),
        ]
        if self.watchdog:
            args.append("--watchdog")
        if self.offline:
            args.append("--offline")
        return args

    model_config = {"extra": "allow"}


# ── Export ────────────────────────────────────────────────────────────────────

class ExportConfig(BaseModel):
    """Mirrors distill_cpp.ExportFormatSpec."""

    format: str     = Field(..., description="Export format key")
    output_dir: str = Field("exports")
    quant_method: str = Field("q4_k_m")
    bits: int       = Field(4, ge=2, le=16)
    group_size: int = Field(128, ge=32, le=512)
    merge_lora: bool = Field(True,  description="Merge LoRA before export")
    push_to_hub: bool = Field(False)
    hub_repo_id: str  = Field("")
    optimize_for: str = Field("balanced", pattern=r"^(speed|quality|size|balanced)$")

    @field_validator("format", mode="before")
    @classmethod
    def normalise_format(cls, v: str) -> str:
        return v.lower().strip()


# ── Hardware snapshot ─────────────────────────────────────────────────────────

class ThermalSnapshot(BaseModel):
    """Mirrors distill_cpp.ThermalReading — used for UI binding."""

    cpu_temp: float   = 0.0
    gpu_temp: float   = 0.0
    soc_temp: float   = 0.0
    cpu_power: float  = 0.0
    gpu_power: float  = 0.0
    total_power: float = 0.0
    available: bool   = False
    error: str        = ""

    def peak_temp(self) -> float:
        return max(self.cpu_temp, self.gpu_temp, self.soc_temp)

    def oom_risk(self, threshold: float = 85.0) -> str:
        p = self.peak_temp()
        if p >= threshold:        return "high"
        if p >= threshold * 0.85: return "medium"
        return "low"


# ── Preference / alignment ────────────────────────────────────────────────────

class AlignmentConfig(BaseModel):
    """DPO / ORPO / SimPO alignment run config."""

    method: Literal["dpo", "orpo", "simpo"] = "orpo"
    model_path: str  = Field(..., min_length=1)
    dataset_path: str = Field("__flywheel__")
    output_dir: str  = Field("outputs/aligned")
    beta: float      = Field(0.1, gt=0, le=1.0)
    epochs: int      = Field(1, ge=1, le=20)
    lr: float        = Field(8e-6, gt=0)
    batch_size: int  = Field(2, ge=1, le=64)
    grad_accum: int  = Field(4, ge=1)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)


# ── Convenience constructors ──────────────────────────────────────────────────

def from_preset(preset_name: str) -> TrainingConfig:
    """Build a TrainingConfig from a named preset."""
    from distill.launch_ui.presets import get_preset
    p = get_preset(preset_name)
    if not p:
        raise ValueError(f"Unknown preset: {preset_name!r}")
    lora_data = {k: p[k] for k in ("lora_rank", "lora_alpha") if k in p}
    lora = LoRAConfig(
        rank=lora_data.get("lora_rank", 16),
        alpha=lora_data.get("lora_alpha", 32),
    )
    return TrainingConfig(
        teacher=p.get("teacher", ""),
        student=p.get("student", ""),
        dataset=p.get("dataset", "yahma/alpaca-cleaned"),
        backend=p.get("backend", "mlx"),
        epochs=int(p.get("epochs", 3)),
        lr=float(p.get("lr", 2e-4)),
        batch_size=int(p.get("batch_size", 4)),
        lora=lora,
    )
