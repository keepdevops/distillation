"""YAML/JSON config editor with Pydantic validation for the Reflex UI.

Provides:
  - ConfigModel: Pydantic model for all distillation config fields
  - validate_config_yaml(): parse + validate YAML/JSON string
  - render_editor(): Gradio-compatible editor (used as fallback in Gradio mode)
"""
from __future__ import annotations

import json
import logging
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ── Config model ───────────────────────────────────────────────────────────────

class DistillConfigModel(BaseModel):
    """Pydantic model for a distillation run configuration."""

    backend: str    = Field("mlx", pattern=r"^(mlx|sft|minillm|unsloth|forward)$")
    teacher: str    = Field(..., min_length=3)
    student: str    = Field(..., min_length=3)
    dataset: str    = Field("yahma/alpaca-cleaned", min_length=3)
    output_dir: str = Field("outputs/distilled", min_length=1)
    epochs: int     = Field(3,  ge=1, le=100)
    lr: float       = Field(2e-4, gt=0, lt=1.0)
    batch_size: int = Field(4,  ge=1, le=128)
    lora_rank: int  = Field(16, ge=4, le=256)
    lora_alpha: int = Field(32, ge=4, le=512)
    max_length: int = Field(512, ge=64, le=8192)
    grad_accum: int = Field(4,  ge=1, le=64)
    warmup_steps: int = Field(100, ge=0)
    export: str     = Field("gguf", pattern=r"^(none|gguf|mlx|coreml|all)$")
    watchdog: bool  = Field(False)
    offline: bool   = Field(False)

    @field_validator("lr", mode="before")
    @classmethod
    def parse_lr(cls, v: Any) -> float:
        if isinstance(v, str):
            return float(v)
        return v

    model_config = {"extra": "allow"}


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_config_yaml(text: str) -> tuple[DistillConfigModel | None, str]:
    """Parse and validate a YAML or JSON config string.

    Returns (model, error_message). error_message is empty on success.
    """
    if not text or not text.strip():
        return None, "Config is empty."
    try:
        # Try YAML first (superset of JSON)
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            return None, f"Expected a mapping, got {type(data).__name__}."
    except yaml.YAMLError as exc:
        return None, f"YAML parse error: {exc}"

    try:
        model = DistillConfigModel(**data)
        return model, ""
    except Exception as exc:
        return None, f"Validation error: {exc}"


def model_to_yaml(model: DistillConfigModel) -> str:
    """Serialise a DistillConfigModel to formatted YAML."""
    data = model.model_dump(exclude_none=True)
    return yaml.dump(data, default_flow_style=False, sort_keys=True, allow_unicode=True)


def model_to_cli_args(model: DistillConfigModel) -> list[str]:
    """Convert config model to CLI argument list for distill.orchestration.agent."""
    args = []
    for field_name, value in model.model_dump().items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{field_name.replace('_', '-')}")
        else:
            args += [f"--{field_name.replace('_', '-')}", str(value)]
    return args


# ── Default templates ──────────────────────────────────────────────────────────

DEFAULT_CONFIG_YAML = """\
# Wow Sausage Maker — distillation config
backend: mlx
teacher: Qwen/Qwen2-1.5B-Instruct
student: Qwen/Qwen2-0.5B-Instruct
dataset: yahma/alpaca-cleaned
output_dir: outputs/distilled
epochs: 3
lr: 0.0002
batch_size: 4
lora_rank: 16
lora_alpha: 32
max_length: 512
grad_accum: 4
warmup_steps: 100
export: gguf
watchdog: false
offline: false
"""

PRESET_CONFIGS: dict[str, str] = {}


def _load_preset_configs() -> None:
    """Populate PRESET_CONFIGS from presets.py at import time."""
    try:
        from distill.launch_ui.presets import PRESETS
        for name, cfg in PRESETS.items():
            subset = {k: v for k, v in cfg.items()
                      if k in DistillConfigModel.model_fields}
            if "teacher" in subset and "student" in subset:
                PRESET_CONFIGS[name] = yaml.dump(subset, default_flow_style=False)
    except Exception:
        pass


_load_preset_configs()


# ── Gradio-compatible editor (used in fallback / dual-UI mode) ─────────────────

def render_editor_gradio() -> None:
    """Render a YAML config editor inside the current gr.Blocks context."""
    import gradio as gr

    gr.Markdown("### YAML Configuration Editor")
    gr.Markdown(
        "Edit the config below. Changes are validated on each keystroke. "
        "Click **Apply** to update the active session."
    )

    with gr.Row():
        preset_dd = gr.Dropdown(
            choices=list(PRESET_CONFIGS.keys()) or ["(no presets)"],
            label="Load Preset",
            scale=1,
        )
        load_btn = gr.Button("Load", variant="secondary", scale=0)

    editor = gr.Code(
        value=DEFAULT_CONFIG_YAML,
        language="yaml",
        label="distill_config.yaml",
        lines=24,
        interactive=True,
    )
    validation_md = gr.Markdown("")
    apply_btn = gr.Button("✅ Apply Config", variant="primary")

    def on_change(text: str) -> str:
        _, err = validate_config_yaml(text)
        return f"❌ {err}" if err else "✅ Valid config"

    def on_load(preset: str) -> tuple[str, str]:
        yaml_text = PRESET_CONFIGS.get(preset, DEFAULT_CONFIG_YAML)
        return yaml_text, "✅ Preset loaded"

    def on_apply(text: str) -> str:
        model, err = validate_config_yaml(text)
        if err:
            return f"❌ Cannot apply: {err}"
        try:
            from distill.ui.state_manager import update_config
            update_config(**model.model_dump())
            return "✅ Config applied to active session"
        except Exception as exc:
            return f"⚠ Applied locally (state_manager unavailable: {exc})"

    editor.change(fn=on_change, inputs=editor, outputs=validation_md)
    load_btn.click(fn=on_load, inputs=preset_dd, outputs=[editor, validation_md])
    apply_btn.click(fn=on_apply, inputs=editor, outputs=validation_md)
