"""CLI mirror component — shows the equivalent CLI command for any UI action.

Usage inside a tab:
    from distill.ui.components.cli_mirror import CliMirror
    mirror = CliMirror()
    mirror.render()
    # Later, update when user changes config:
    mirror.update(backend="mlx", teacher="Qwen2.5-7B", student="Qwen2.5-1.5B", ...)
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

import gradio as gr


@dataclass
class CliMirrorState:
    """Holds the current set of CLI args to render."""
    command: str = "python -m distill.orchestration.agent"
    flags: dict[str, Any] = field(default_factory=dict)

    def to_cmd(self) -> str:
        parts = [self.command]
        for k, v in self.flags.items():
            if v is None or v == "":
                continue
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    parts.append(flag)
            else:
                parts.append(f"{flag} {shlex.quote(str(v))}")
        return " \\\n    ".join(parts)


class CliMirror:
    """Gradio component that renders a live CLI mirror box."""

    def __init__(self, label: str = "Equivalent CLI Command") -> None:
        self.label = label
        self._state = CliMirrorState()
        self._box: gr.Code | None = None

    def render(self) -> gr.Code:
        """Render the CLI mirror code block. Call inside a gr.Blocks context."""
        with gr.Accordion(f"🖥 {self.label}", open=False):
            gr.Markdown(
                "<small>Copy this command to reproduce this run headlessly "
                "or schedule it via cron.</small>",
            )
            self._box = gr.Code(
                value=self._state.to_cmd(),
                language="shell",
                label="",
                interactive=False,
                elem_classes=["cli-mirror"],
            )
        return self._box

    def update(self, **flags: Any) -> str:
        """Update CLI flags and return the new command string.

        Wire this to a gr.State update or .change() handler:
            backend_dd.change(fn=mirror.update, inputs=[backend_dd], outputs=[mirror.box])
        """
        self._state.flags.update(flags)
        cmd = self._state.to_cmd()
        if self._box is not None:
            return gr.update(value=cmd)
        return cmd

    @property
    def box(self) -> gr.Code | None:
        return self._box


def build_cli_mirror_for_config(
    teacher: str,
    student: str,
    backend: str,
    dataset: str,
    epochs: int,
    lr: float,
    batch_size: int,
    output_dir: str,
) -> str:
    """Convenience builder — returns a CLI string from common training params."""
    state = CliMirrorState(
        flags={
            "teacher": teacher,
            "student": student,
            "backend": backend,
            "dataset": dataset,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "output_dir": output_dir,
        }
    )
    return state.to_cmd()
