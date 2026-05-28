"""Preset model/dataset IDs and named configuration galleries for the launcher."""
from __future__ import annotations

from typing import Any

KNOWN_TEACHERS = [
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "ibm-granite/granite-3.1-8b-instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-2-9b-it",
]

KNOWN_STUDENTS = [
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]

KNOWN_DATASETS = [
    "tatsu-lab/alpaca",
    "yahma/alpaca-cleaned",
    "mlabonne/guanaco-llama2-1k",
    "HuggingFaceH4/no_robots",
    "teknium/OpenHermes-2.5",
    "Magpie-Align/Magpie-Pro-300K-Filtered",
    "argilla/distilabel-capybara-dpo-7k-binarized",
    "bigcode/self-oss-instruct-sc2-exec-filter-50k",
    "m-a-p/CodeFeedback-Filtered-Instruction",
    "gsm8k",
]

# ── Named configuration presets ───────────────────────────────────────────────
# Each preset is a dict of kwargs matching distill.orchestration.agent CLI flags.

PRESETS: dict[str, dict[str, Any]] = {
    "DevOps Agent": {
        "description": "Fast, practical assistant for shell, CI/CD, and infra tasks.",
        "teacher":     "Qwen/Qwen2-7B-Instruct",
        "student":     "Qwen/Qwen2-1.5B-Instruct",
        "dataset":     "yahma/alpaca-cleaned",
        "backend":     "mlx",
        "epochs":      3,
        "lr":          2e-4,
        "batch_size":  4,
        "lora_rank":   16,
        "system_prompt": (
            "You are a DevOps engineer. Answer concisely with working shell commands, "
            "Docker, Kubernetes, and CI/CD examples."
        ),
        "icon": "🔧",
    },
    "Code Specialist": {
        "description": "Deep code generation and review — Python, Rust, TypeScript.",
        "teacher":     "meta-llama/Llama-3.2-8B-Instruct",
        "student":     "meta-llama/Llama-3.2-1B-Instruct",
        "dataset":     "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "backend":     "mlx",
        "epochs":      4,
        "lr":          1e-4,
        "batch_size":  2,
        "lora_rank":   32,
        "system_prompt": (
            "You are an expert software engineer. Write clean, tested, idiomatic code. "
            "Explain your reasoning briefly."
        ),
        "icon": "💻",
    },
    "Math Reasoner": {
        "description": "Step-by-step mathematical reasoning and problem solving.",
        "teacher":     "Qwen/Qwen2-7B-Instruct",
        "student":     "Qwen/Qwen2-0.5B-Instruct",
        "dataset":     "gsm8k",
        "backend":     "mlx",
        "epochs":      5,
        "lr":          3e-4,
        "batch_size":  4,
        "lora_rank":   16,
        "system_prompt": (
            "You are a mathematics tutor. Think step by step, show your work, "
            "and verify your answer."
        ),
        "icon": "🔢",
    },
    "Fast Edge (Q4)": {
        "description": "Optimised for edge/CPU deployment. Smallest footprint, Q4 GGUF.",
        "teacher":     "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "student":     "HuggingFaceTB/SmolLM2-135M-Instruct",
        "dataset":     "yahma/alpaca-cleaned",
        "backend":     "sft",
        "epochs":      2,
        "lr":          5e-4,
        "batch_size":  8,
        "lora_rank":   8,
        "quant_method": "q4_k_m",
        "system_prompt": "You are a helpful assistant.",
        "icon": "⚡",
    },
    "Research Assistant": {
        "description": "Long-context summarisation and analysis of technical documents.",
        "teacher":     "google/gemma-2-9b-it",
        "student":     "microsoft/Phi-3-mini-4k-instruct",
        "dataset":     "HuggingFaceH4/no_robots",
        "backend":     "mlx",
        "epochs":      3,
        "lr":          2e-4,
        "batch_size":  2,
        "lora_rank":   32,
        "system_prompt": (
            "You are a research assistant. Summarise, analyse, and explain technical "
            "content clearly and accurately."
        ),
        "icon": "📚",
    },
}


def preset_names() -> list[str]:
    return list(PRESETS.keys())


def get_preset(name: str) -> dict[str, Any]:
    """Return a preset dict by name, or empty dict if not found."""
    return PRESETS.get(name, {})


def preset_choices_html() -> str:
    """Return HTML cards for all presets (for use in gr.HTML)."""
    cards = []
    for name, cfg in PRESETS.items():
        icon = cfg.get("icon", "🤖")
        desc = cfg.get("description", "")
        cards.append(
            f'<div style="background:#1e293b;border-radius:.5rem;padding:.75rem 1rem;'
            f'margin:.3rem;min-width:180px;max-width:220px;display:inline-block">'
            f'  <div style="font-size:1.4rem">{icon}</div>'
            f'  <div style="font-weight:700;color:#e2e8f0;margin:.2rem 0">{name}</div>'
            f'  <div style="font-size:.75rem;color:#94a3b8">{desc}</div>'
            f'</div>'
        )
    return '<div style="display:flex;flex-wrap:wrap;gap:.5rem">' + "".join(cards) + "</div>"
