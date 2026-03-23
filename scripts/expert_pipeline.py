#!/usr/bin/env python3
"""
Expert distillation pipeline for domain-specific models (tax, legal, medical, etc.)

Stages:
  inspect  — print dataset columns + sample rows as JSON (for Gradio UI)
  remap    — load HF dataset, remap columns → instruction/input/output, save locally
  cot      — generate Chain-of-Thought rationales using a GGUF teacher via llama-server
  distill  — launch run_distillation_agent.py on the generated data

Usage:
  python scripts/expert_pipeline.py --mode inspect --dataset nelson-liu/legalbench

  python scripts/expert_pipeline.py --mode remap \\
      --dataset Atome-LLM/Tax-Policy-Analysis \\
      --instruction_col question --output_col answer \\
      --output_dir ./domain_data/tax_expert --max_samples 5000

  python scripts/expert_pipeline.py --mode cot \\
      --dataset ./domain_data/tax_expert \\
      --teacher /Users/Shared/llama/models/phi-4-Q5_K_M.gguf \\
      --domain tax --n_samples 1000 \\
      --output_dir ./domain_data/tax_cot

  python scripts/expert_pipeline.py --mode distill \\
      --dataset ./domain_data/tax_cot \\
      --output_dir ./runs/tax-expert \\
      --backend mlx --open
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPTS_DIR.parent

sys.path.insert(0, str(SCRIPTS_DIR))

GGUF_MODEL_DIR = Path("/Users/Shared/llama/models")

# ── Domain CoT system prompts ──────────────────────────────────────────────────

DOMAIN_SYSTEM_PROMPTS: dict[str, str] = {
    "tax": (
        "You are a precise US tax law reasoning assistant. "
        "When answering questions:\n"
        "1. Reason step-by-step through the relevant IRC section, regulation, or ruling\n"
        "2. Cite specific code sections, thresholds, phase-out ranges, and tax year\n"
        "3. Flag numerical constants (dollar amounts, percentages, dates) explicitly\n"
        "4. Provide a clear, definitive final answer\n\n"
        "Format your response as:\n"
        "<reasoning>\n[step-by-step analysis]\n</reasoning>\n"
        "<answer>\n[final answer]\n</answer>"
    ),
    "legal": (
        "You are a precise legal reasoning assistant. "
        "When answering questions:\n"
        "1. Identify the controlling statute, regulation, or case law\n"
        "2. Apply the relevant legal test or standard step-by-step\n"
        "3. Note any jurisdiction-specific variations or exceptions\n"
        "4. Provide a clear conclusion\n\n"
        "Format your response as:\n"
        "<reasoning>\n[legal analysis]\n</reasoning>\n"
        "<answer>\n[conclusion]\n</answer>"
    ),
    "medical": (
        "You are a precise medical reasoning assistant. "
        "When answering questions:\n"
        "1. Apply clinical reasoning step-by-step\n"
        "2. Reference relevant guidelines, dosing thresholds, or diagnostic criteria\n"
        "3. Note contraindications or edge cases\n"
        "4. Provide a clear clinical recommendation\n\n"
        "Format your response as:\n"
        "<reasoning>\n[clinical analysis]\n</reasoning>\n"
        "<answer>\n[recommendation]\n</answer>"
    ),
    "finance": (
        "You are a precise financial reasoning assistant. "
        "When answering questions:\n"
        "1. Apply relevant financial concepts, formulas, or regulations step-by-step\n"
        "2. Use exact figures, rates, and regulatory thresholds\n"
        "3. Note risk factors or assumptions\n"
        "4. Provide a clear, quantified answer where possible\n\n"
        "Format your response as:\n"
        "<reasoning>\n[financial analysis]\n</reasoning>\n"
        "<answer>\n[conclusion]\n</answer>"
    ),
    "coding": (
        "You are a precise programming assistant. "
        "When answering questions:\n"
        "1. Reason through the problem step-by-step\n"
        "2. Consider edge cases, complexity, and correctness\n"
        "3. Explain key design decisions\n"
        "4. Provide working, well-commented code\n\n"
        "Format your response as:\n"
        "<reasoning>\n[analysis and approach]\n</reasoning>\n"
        "<answer>\n[code and explanation]\n</answer>"
    ),
}
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise reasoning assistant. "
    "Reason step-by-step through the question, then provide a clear final answer.\n\n"
    "Format your response as:\n"
    "<reasoning>\n[analysis]\n</reasoning>\n"
    "<answer>\n[answer]\n</answer>"
)

# ── Column auto-detection heuristics ──────────────────────────────────────────

_INSTRUCTION_CANDIDATES = [
    "instruction", "question", "query", "prompt", "input", "text",
    "problem", "task", "request",
]
_OUTPUT_CANDIDATES = [
    "output", "answer", "response", "completion", "target", "label",
    "solution", "reply",
]
_INPUT_CANDIDATES = [
    "input", "context", "passage", "document", "background",
]


def _guess_columns(cols: list[str]) -> dict[str, str | None]:
    """Heuristically map dataset columns to instruction/input/output."""
    cols_lower = {c.lower(): c for c in cols}
    def pick(candidates):
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        return None

    instruction = pick(_INSTRUCTION_CANDIDATES)
    output = pick(_OUTPUT_CANDIDATES)
    # input/context should not overlap with instruction
    input_col = None
    for c in _INPUT_CANDIDATES:
        if c in cols_lower and cols_lower[c] != instruction:
            input_col = cols_lower[c]
            break
    return {"instruction": instruction, "output": output, "input": input_col}


# ── Stage: inspect ─────────────────────────────────────────────────────────────

def cmd_inspect(args: argparse.Namespace) -> None:
    """Load dataset, print columns + preview rows as JSON to stdout."""
    ds = _load_any_dataset(args.dataset)

    cols = list(ds.column_names)
    guessed = _guess_columns(cols)
    preview = [dict(row) for row in ds.select(range(min(3, len(ds))))]
    # Truncate long values for display
    for row in preview:
        for k, v in row.items():
            if isinstance(v, str) and len(v) > 200:
                row[k] = v[:200] + "…"

    result = {
        "columns": cols,
        "n_rows": len(ds),
        "guessed": guessed,
        "preview": preview,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ── Stage: remap ──────────────────────────────────────────────────────────────

def cmd_remap(args: argparse.Namespace) -> None:
    """Remap columns and save as local HF dataset."""
    ds = _load_any_dataset(args.dataset)

    if args.max_samples and len(ds) > args.max_samples:
        LOG.info("Truncating to %d samples", args.max_samples)
        ds = ds.select(range(args.max_samples))

    instruction_col = args.instruction_col
    output_col = args.output_col
    input_col = getattr(args, "input_col", None) or None

    missing = [c for c in [instruction_col, output_col] if c and c not in ds.column_names]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}. Available: {ds.column_names}")

    def remap(row):
        out = {
            "instruction": str(row[instruction_col]) if instruction_col else "",
            "output": str(row[output_col]) if output_col else "",
            "input": str(row[input_col]) if input_col and input_col in row else "",
        }
        return out

    LOG.info("Remapping columns: instruction=%s  output=%s  input=%s",
             instruction_col, output_col, input_col)
    remapped = ds.map(remap, remove_columns=ds.column_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remapped.save_to_disk(str(output_dir))
    LOG.info("Saved %d rows → %s", len(remapped), output_dir)

    # Also write a jsonl for easy inspection
    jsonl_path = output_dir / "remapped.jsonl"
    with open(jsonl_path, "w") as f:
        for row in remapped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOG.info("JSONL copy → %s", jsonl_path)


# ── Stage: cot ────────────────────────────────────────────────────────────────

def _load_any_dataset(dataset_arg: str):
    """Load a dataset from: HF disk dir, JSONL file, or HF Hub ID."""
    from datasets import load_from_disk, load_dataset as hf_load

    p = Path(dataset_arg)

    # Resolve relative paths from project root if not found from CWD
    if not p.is_absolute() and not p.exists():
        alt = SCRIPTS_DIR.parent / dataset_arg.lstrip("./")
        if alt.exists():
            p = alt

    if p.exists():
        if p.is_dir():
            # HF dataset saved to disk — check for hf_dataset subdir
            hf_sub = p / "hf_dataset"
            load_path = hf_sub if hf_sub.exists() else p
            LOG.info("Loading HF disk dataset: %s", load_path)
            ds = load_from_disk(str(load_path))
        elif p.suffix in (".jsonl", ".json"):
            LOG.info("Loading JSONL: %s", p)
            ds = hf_load("json", data_files=str(p), split="train")
        else:
            raise ValueError(f"Unsupported local file type: {p}")
    else:
        # Assume HF Hub ID
        LOG.info("Loading HF Hub dataset: %s", dataset_arg)
        ds = hf_load(dataset_arg, split="train", trust_remote_code=True)

    if hasattr(ds, "keys") and "train" in ds:
        ds = ds["train"]
    return ds


def cmd_cot(args: argparse.Namespace) -> None:
    """Generate Chain-of-Thought rationales using a GGUF teacher via llama-server."""
    from cpp_eval_utils import LlamaServer

    LOG.info("Loading dataset: %s", args.dataset)
    ds = _load_any_dataset(args.dataset)
    if hasattr(ds, "keys") and "train" in ds:
        ds = ds["train"]

    n = min(args.n_samples, len(ds))
    LOG.info("Generating CoT for %d samples (teacher: %s)", n, args.teacher)
    ds = ds.select(range(n))

    domain = getattr(args, "domain", None) or "general"
    system_prompt = (
        getattr(args, "system_prompt", None)
        or DOMAIN_SYSTEM_PROMPTS.get(domain, DEFAULT_SYSTEM_PROMPT)
    )
    LOG.info("Domain: %s", domain)

    temperature = getattr(args, "temperature", 0.3)
    max_tokens = getattr(args, "max_tokens", 1024)
    ctx_size = getattr(args, "ctx_size", 8192)
    n_parallel = getattr(args, "n_parallel", 4)
    batch_size = getattr(args, "batch_size", 16)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_dir / "cot_data.jsonl"

    def _make_prompt(row: dict) -> str:
        instruction = row.get("instruction", "")
        context = row.get("input", "").strip()
        if context:
            return f"{system_prompt}\n\nContext: {context}\n\nQuestion: {instruction}"
        return f"{system_prompt}\n\nQuestion: {instruction}"

    prompts = [_make_prompt(dict(row)) for row in ds]

    req_timeout = max(120, int(max_tokens * 0.6) + 90)
    LOG.info("Request timeout per completion: %ds  (max_tokens=%d)", req_timeout, max_tokens)

    written = 0
    with LlamaServer(
        gguf_path=args.teacher,
        ctx_size=ctx_size,
        n_parallel=n_parallel,
    ) as srv:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with open(out_jsonl, "w") as jf:
            for batch_start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[batch_start: batch_start + batch_size]
                batch_rows = list(ds.select(range(batch_start, batch_start + len(batch_prompts))))

                with ThreadPoolExecutor(max_workers=n_parallel) as pool:
                    futures = {
                        pool.submit(srv.complete, p, max_tokens, temperature, req_timeout): i
                        for i, p in enumerate(batch_prompts)
                    }
                    from concurrent.futures import as_completed as _ac
                    batch_results: dict[int, str] = {}
                    for fut in _ac(futures):
                        idx = futures[fut]
                        try:
                            batch_results[idx] = fut.result()
                        except Exception as e:
                            LOG.warning("CoT generation failed for sample %d: %s", batch_start + idx, e)
                            batch_results[idx] = ""

                for i, row in enumerate(batch_rows):
                    cot_text = batch_results.get(i, "")
                    if not cot_text.strip():
                        continue
                    entry = {
                        "instruction": row.get("instruction", ""),
                        "input": row.get("input", ""),
                        "output": cot_text.strip(),
                        "original_output": row.get("output", ""),
                        "domain": domain,
                    }
                    jf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    written += 1

                LOG.info("CoT progress: %d/%d  (written=%d)", batch_start + len(batch_prompts), n, written)

    LOG.info("CoT generation complete: %d samples → %s", written, out_jsonl)

    # Convert JSONL to HF dataset for direct use in distillation
    from datasets import Dataset as HFDataset
    rows = []
    with open(out_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    hf_ds = HFDataset.from_list(rows)
    hf_ds.save_to_disk(str(output_dir / "hf_dataset"))
    LOG.info("HF dataset saved → %s/hf_dataset (%d rows)", output_dir, len(hf_ds))


# ── Stage: distill ────────────────────────────────────────────────────────────

def cmd_distill(args: argparse.Namespace) -> None:
    """Launch run_distillation_agent.py on the expert dataset."""
    import subprocess

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "run_distillation_agent.py"),
        "--dataset", args.dataset,
        "--output_dir", args.output_dir,
        "--backend", getattr(args, "backend", "mlx"),
        "--epochs", str(getattr(args, "epochs", 3)),
        "--max_samples", str(getattr(args, "max_samples", 3000)),
        "--lora_r", str(getattr(args, "lora_r", 32)),
        "--export", getattr(args, "export", "gguf"),
        "--benchmarks",
        "--log_experiment",
    ]
    # In offline/air-gapped mode, force --open (Qwen2 models in airgap_bundle).
    # Gated models like Llama-3.2-8B require HF auth which is unavailable offline.
    if getattr(args, "open", False) or getattr(args, "offline", False):
        cmd.append("--open")
    if getattr(args, "offline", False):
        cmd.append("--offline")

    LOG.info("Launching distillation: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    sys.exit(result.returncode)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Expert distillation pipeline")
    ap.add_argument("--mode", required=True,
                    choices=["inspect", "remap", "cot", "distill"],
                    help="Pipeline stage to run")
    ap.add_argument("--dataset", type=str, required=True,
                    help="HF dataset ID or local path")
    ap.add_argument("--output_dir", type=str, default="./domain_data/expert")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Max samples to load (0 = all)")
    # remap
    ap.add_argument("--instruction_col", type=str, default="instruction")
    ap.add_argument("--output_col", type=str, default="output")
    ap.add_argument("--input_col", type=str, default="")
    # cot
    ap.add_argument("--teacher", type=str, default="",
                    help="Path to GGUF teacher model")
    ap.add_argument("--domain", type=str, default="general",
                    choices=list(DOMAIN_SYSTEM_PROMPTS.keys()) + ["general"])
    ap.add_argument("--system_prompt", type=str, default="",
                    help="Override CoT system prompt")
    ap.add_argument("--n_samples", type=int, default=1000,
                    help="Number of CoT samples to generate")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--ctx_size", type=int, default=8192)
    ap.add_argument("--n_parallel", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    # distill
    ap.add_argument("--backend", type=str, default="mlx",
                    choices=["mlx", "pytorch", "unsloth"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--export", type=str, default="gguf")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--offline", action="store_true")

    args = ap.parse_args()

    if args.mode == "inspect":
        cmd_inspect(args)
    elif args.mode == "remap":
        cmd_remap(args)
    elif args.mode == "cot":
        cmd_cot(args)
    elif args.mode == "distill":
        cmd_distill(args)


if __name__ == "__main__":
    main()
