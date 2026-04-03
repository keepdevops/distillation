"""Gradio layout for the distillation launcher."""
from __future__ import annotations

import gradio as gr

from .discovery import (
    discover_datasets,
    discover_output_dirs,
    discover_students,
    discover_teachers,
)
from ..infra.paths import project_dir
from .tabs import (
    build_tab_configure,
    build_tab_data_prep,
    build_tab_domain,
    build_tab_eval,
    build_tab_expert,
    build_tab_logs,
    build_tab_help,
    wire_events,
)

PROJECT_DIR = project_dir()


def build_ui():
    # Discover once — reused by all tabs to avoid repeated filesystem scans.
    teachers = discover_teachers()
    students = discover_students()
    datasets = discover_datasets()
    out_dirs = discover_output_dirs()

    default_teacher = "Qwen/Qwen2-1.5B-Instruct"
    default_student = students[0] if students else "Qwen/Qwen2-0.5B-Instruct"
    default_dataset = "yahma/alpaca-cleaned"
    default_out     = str(PROJECT_DIR / "distilled-minillm")

    defaults = {
        "default_teacher": default_teacher,
        "default_student": default_student,
        "default_dataset": default_dataset,
        "default_out":     default_out,
    }

    with gr.Blocks(title="Distillation Launcher") as demo:
        gr.Markdown("# Distillation Launcher")

        with gr.Tabs():
            cfg_widgets    = build_tab_configure(teachers, students, datasets,
                                                 out_dirs, defaults)
            prep_widgets   = build_tab_data_prep(teachers, datasets)
            domain_widgets = build_tab_domain(teachers)
            eval_widgets   = build_tab_eval(
                teachers, students, datasets, out_dirs,
                default_teacher, default_student, default_dataset, default_out,
            )
            expert_widgets = build_tab_expert(students)
            logs_widgets   = build_tab_logs()
            _              = build_tab_help()

        # Merge all widget dicts into one flat namespace for wiring.
        all_widgets = {
            **cfg_widgets,
            **prep_widgets,
            **domain_widgets,
            **eval_widgets,
            **expert_widgets,
            **logs_widgets,
        }

        wire_events(demo, all_widgets)

    return demo
