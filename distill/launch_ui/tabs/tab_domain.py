"""Domain Synthesis tab widget layout."""
from __future__ import annotations

import gradio as gr
from pathlib import Path

from ...infra.paths import project_dir

PROJECT_DIR = project_dir()

_EP_HF_DATASETS = [
    # Legal
    "nguha/legalbench",
    "nelson-liu/legalbench",
    # Tax
    "Atome-LLM/Tax-Policy-Analysis",
    # Medical
    "medalpaca/medical_meadow_medical_flashcards",
    "medalpaca/medical_meadow_wikidoc",
    "pubmed_qa",
    # Finance
    "gbharti/finance-alpaca",
    "FinGPT/fingpt-sentiment-train",
    # Coding
    "iamtarun/python_code_instructions_18k_alpaca",
    "sahil2801/CodeAlpaca-20k",
    # General
    "yahma/alpaca-cleaned",
    "tatsu-lab/alpaca",
    "HuggingFaceH4/ultrachat_200k",
]

_EP_LOCAL_CANDIDATES = [
    "./domain_data/expert_remapped",
    "./domain_data/tax",
    "./domain_data/legal",
    "./domain_data/medical",
    "./domain_data/coding",
]


def _ep_dataset_choices():
    local = [p for p in _EP_LOCAL_CANDIDATES
             if (PROJECT_DIR / p.lstrip("./")).exists()]
    return _EP_HF_DATASETS + local


def build_tab_domain(teachers):
    """Build the Domain Synthesis tab.

    Parameters
    ----------
    teachers:  list of teacher model choices

    Returns
    -------
    dict of all domain synthesis widgets required for event wiring.
    """
    with gr.Tab("Domain Synthesis"):
        gr.Markdown("# Specialized Domain Synthesis")
        gr.Markdown(
            "Generate high-quality domain-focused instruction-response pairs using "
            "Magpie-style self-synthesis. Each domain uses curated system prompts and "
            "domain-appropriate quality filters (e.g. coding requires code blocks, "
            "math/tax require numbers). Output is an HF dataset ready for distillation."
        )

        # ── Shared domain controls ────────────────────────────────────────
        with gr.Row():
            dom_teacher = gr.Dropdown(
                choices=teachers,
                value="Qwen/Qwen2-1.5B-Instruct",
                label="Teacher model",
                allow_custom_value=True,
                scale=4,
                info="Larger teacher = richer domain knowledge. Qwen2-1.5B works well for all domains.",
            )
            dom_refresh_btn = gr.Button("Refresh", scale=1, size="sm")
        with gr.Row():
            dom_backend = gr.Radio(
                ["auto", "mlx", "mps"], value="auto",
                label="Backend",
                info="auto picks MLX on Apple Silicon (2-4× faster than MPS)",
            )
            dom_offline = gr.Checkbox(value=False, label="Offline (use cached model)")
            dom_filter  = gr.Checkbox(value=True,  label="Deep filter output (dedup + quality)")
            dom_batch   = gr.Slider(1, 64, value=32, step=1, label="Batch size",
                                    info="For MLX: loop chunk size only (sequential gen). For MPS: reduce to 4-8 if OOM.")

        dom_status      = gr.Textbox(value="idle", label="Status", interactive=False)
        domain_progress = gr.HTML(value="")

        gr.Markdown("---")

        # ── Medical ───────────────────────────────────────────────────────
        with gr.Accordion("🏥  Medical", open=False):
            gr.Markdown(
                "Generates clinical education Q&A: symptoms, diagnosis reasoning, pharmacology, "
                "pathophysiology, anatomy, public health, and medical ethics. "
                "**Filter:** responses ≥40 words, distinct-2 ≥0.30. "
                "Output tagged `domain=medical` for easy mixing with other datasets."
            )
            with gr.Row():
                med_n      = gr.Slider(500, 20000, value=3000, step=500,
                                       label="Pairs to generate (before filter)")
                med_target = gr.Slider(200, 10000, value=1500, step=200,
                                       label="Target keep after filter")
                med_outdir = gr.Textbox(
                    value=str(PROJECT_DIR / "domain_data" / "medical"),
                    label="Output directory", scale=3,
                )
            med_btn = gr.Button("Generate Medical Dataset", variant="primary")

        # ── Math ──────────────────────────────────────────────────────────
        with gr.Accordion("📐  Mathematics", open=False):
            gr.Markdown(
                "Generates math problem-solution pairs: calculus, linear algebra, statistics, "
                "discrete math, number theory, and competition math. "
                "**Filter:** response must contain at least one number/equation, distinct-2 ≥0.25."
            )
            with gr.Row():
                math_n      = gr.Slider(500, 20000, value=3000, step=500,
                                        label="Pairs to generate (before filter)")
                math_target = gr.Slider(200, 10000, value=1500, step=200,
                                        label="Target keep after filter")
                math_outdir = gr.Textbox(
                    value=str(PROJECT_DIR / "domain_data" / "math"),
                    label="Output directory", scale=3,
                )
            math_btn = gr.Button("Generate Math Dataset", variant="primary")

        # ── Legal ─────────────────────────────────────────────────────────
        with gr.Accordion("⚖️  Legal / Law", open=False):
            gr.Markdown(
                "Generates legal education Q&A: contracts, torts, constitutional law, criminal law, "
                "property, corporate, IP, civil procedure, and legal reasoning. "
                "**Filter:** responses ≥50 words (legal explanations are inherently longer), distinct-2 ≥0.30. "
                "All outputs include an educational disclaimer."
            )
            with gr.Row():
                legal_n      = gr.Slider(500, 20000, value=3000, step=500,
                                         label="Pairs to generate (before filter)")
                legal_target = gr.Slider(200, 10000, value=1500, step=200,
                                         label="Target keep after filter")
                legal_outdir = gr.Textbox(
                    value=str(PROJECT_DIR / "domain_data" / "legal"),
                    label="Output directory", scale=3,
                )
            legal_btn = gr.Button("Generate Legal Dataset", variant="primary")

        # ── Tax ───────────────────────────────────────────────────────────
        with gr.Accordion("🧾  Tax / IRS", open=False):
            gr.Markdown(
                "Generates U.S. tax education Q&A: individual filing, business taxes, capital gains, "
                "retirement accounts, self-employment, SALT, estate/gift tax, and audits. "
                "**Filter:** responses ≥40 words, must contain numbers (tax rules involve figures), "
                "distinct-2 ≥0.28."
            )
            with gr.Row():
                tax_n      = gr.Slider(500, 20000, value=3000, step=500,
                                       label="Pairs to generate (before filter)")
                tax_target = gr.Slider(200, 10000, value=1500, step=200,
                                       label="Target keep after filter")
                tax_outdir = gr.Textbox(
                    value=str(PROJECT_DIR / "domain_data" / "tax"),
                    label="Output directory", scale=3,
                )
            tax_btn = gr.Button("Generate Tax Dataset", variant="primary")

        # ── Coding ────────────────────────────────────────────────────────
        with gr.Accordion("💻  Coding / Programming", open=False):
            gr.Markdown(
                "Generates programming Q&A with working code: Python, JavaScript, C/C++, Rust, Go, "
                "SQL, shell scripting, algorithms, ML engineering, DevOps, and API design. "
                "**Filter:** response must contain at least one code fence (``` block or inline `code`), "
                "distinct-2 ≥0.20 (code is naturally repetitive)."
            )
            with gr.Row():
                code_n      = gr.Slider(500, 20000, value=4000, step=500,
                                        label="Pairs to generate (before filter)",
                                        info="Generate more — code filter is stricter (~30-40% pass rate)")
                code_target = gr.Slider(200, 10000, value=2000, step=200,
                                        label="Target keep after filter")
                code_outdir = gr.Textbox(
                    value=str(PROJECT_DIR / "domain_data" / "coding"),
                    label="Output directory", scale=3,
                )
            code_btn = gr.Button("Generate Coding Dataset", variant="primary")

        # ── Finance ───────────────────────────────────────────────────────
        with gr.Accordion("💰  Finance / Investing", open=False):
            gr.Markdown(
                "Generates finance education Q&A: personal finance, investing, corporate finance, "
                "derivatives, fixed income, macroeconomics, risk management, and financial modeling. "
                "**Filter:** responses ≥30 words, must contain numbers (finance involves figures), "
                "distinct-2 ≥0.28."
            )
            with gr.Row():
                fin_n      = gr.Slider(500, 20000, value=3000, step=500,
                                       label="Pairs to generate (before filter)")
                fin_target = gr.Slider(200, 10000, value=1500, step=200,
                                       label="Target keep after filter")
                fin_outdir = gr.Textbox(
                    value=str(PROJECT_DIR / "domain_data" / "finance"),
                    label="Output directory", scale=3,
                )
            fin_btn = gr.Button("Generate Finance Dataset", variant="primary")

        # ── Custom Domain Builder ─────────────────────────────────────────
        with gr.Accordion("✏️  Custom Domain Builder", open=False):
            gr.Markdown(
                "Define a new domain with custom system prompts and quality filters. "
                "**Save** writes it to `configs/domain_prompts.json` so it persists across sessions "
                "and is immediately available to all domain synthesis tools."
            )
            with gr.Row():
                custom_id    = gr.Textbox(label="Domain ID",
                                          placeholder="e.g. chemistry  (lowercase, no spaces)",
                                          scale=2)
                custom_label = gr.Textbox(label="Display label",
                                          placeholder="e.g. Chemistry / Science",
                                          scale=3)
            custom_desc = gr.Textbox(
                label="Description (shown in UI, optional)",
                placeholder="Chemistry Q&A: reactions, periodic table, organic/inorganic, lab safety.",
                lines=2,
            )
            custom_prompts = gr.Textbox(
                label="System prompts — one per line (at least 1 required)",
                placeholder=(
                    "You are a chemistry educator explaining reactions clearly.\n"
                    "You are an organic chemistry tutor covering nomenclature and mechanisms.\n"
                    "You are a physical chemistry instructor discussing thermodynamics and kinetics."
                ),
                lines=6,
            )
            gr.Markdown("**Quality filter settings**")
            with gr.Row():
                custom_min_words  = gr.Slider(5, 100, value=20, step=5,
                                              label="Min response words")
                custom_max_words  = gr.Slider(100, 2000, value=600, step=50,
                                              label="Max response words")
                custom_min_d2     = gr.Slider(0.10, 0.60, value=0.30, step=0.05,
                                              label="Min distinct-2")
            with gr.Row():
                custom_req_code   = gr.Checkbox(value=False,
                                                label="Require code block (``` or inline `code`)")
                custom_req_nums   = gr.Checkbox(value=False,
                                                label="Require numbers in response")
            custom_save_status = gr.Textbox(value="", label="Save status", interactive=False)
            with gr.Row():
                custom_save_btn = gr.Button("Save Domain", variant="secondary", scale=2)

            gr.Markdown("**Generate with custom domain** (saves first if domain is new)")
            with gr.Row():
                custom_n      = gr.Slider(500, 20000, value=3000, step=500,
                                          label="Pairs to generate")
                custom_target = gr.Slider(200, 10000, value=1500, step=200,
                                          label="Target keep after filter")
                custom_outdir = gr.Textbox(
                    value=str(PROJECT_DIR / "domain_data" / "custom"),
                    label="Output directory", scale=3,
                )
            custom_launch_btn = gr.Button("Save & Generate", variant="primary")

        gr.Markdown("---")
        gr.Markdown(
            "### After generation\n"
            "Point the **Dataset** field in **Configure & Launch** at the output directory "
            "(e.g. `domain_data/coding/hf_dataset`) and launch distillation normally. "
            "You can also merge multiple domain datasets using `datasets.concatenate_datasets()` "
            "before distillation for a multi-domain specialist model."
        )

    return {
        "dom_teacher": dom_teacher,
        "dom_refresh_btn": dom_refresh_btn,
        "dom_backend": dom_backend,
        "dom_offline": dom_offline,
        "dom_filter": dom_filter,
        "dom_batch": dom_batch,
        "dom_status": dom_status,
        "domain_progress": domain_progress,
        "med_n": med_n,
        "med_target": med_target,
        "med_outdir": med_outdir,
        "med_btn": med_btn,
        "math_n": math_n,
        "math_target": math_target,
        "math_outdir": math_outdir,
        "math_btn": math_btn,
        "legal_n": legal_n,
        "legal_target": legal_target,
        "legal_outdir": legal_outdir,
        "legal_btn": legal_btn,
        "tax_n": tax_n,
        "tax_target": tax_target,
        "tax_outdir": tax_outdir,
        "tax_btn": tax_btn,
        "code_n": code_n,
        "code_target": code_target,
        "code_outdir": code_outdir,
        "code_btn": code_btn,
        "fin_n": fin_n,
        "fin_target": fin_target,
        "fin_outdir": fin_outdir,
        "fin_btn": fin_btn,
        "custom_id": custom_id,
        "custom_label": custom_label,
        "custom_desc": custom_desc,
        "custom_prompts": custom_prompts,
        "custom_min_words": custom_min_words,
        "custom_max_words": custom_max_words,
        "custom_min_d2": custom_min_d2,
        "custom_req_code": custom_req_code,
        "custom_req_nums": custom_req_nums,
        "custom_save_status": custom_save_status,
        "custom_save_btn": custom_save_btn,
        "custom_n": custom_n,
        "custom_target": custom_target,
        "custom_outdir": custom_outdir,
        "custom_launch_btn": custom_launch_btn,
    }
