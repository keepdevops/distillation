#!/usr/bin/env python3
"""
Magpie-style self-synthesis: auto-generate (instruction, response) pairs
from a teacher model by conditioning on the chat-template user-turn prefix.

Reference: "Magpie: Alignment Data Synthesis from Scratch by Conditioning
LLMs on Their Instruction-Following Demonstrations" (Xu et al., 2024)

How it works:
  1. Feed the model just the opening of a user turn:
       <|im_start|>system\\n{sys_prompt}<|im_end|>\\n<|im_start|>user\\n
  2. The model generates what it "thinks" a user would ask  →  instruction
  3. Append <|im_end|>\\n<|im_start|>assistant\\n and generate  →  response
  4. Filter inline (length, refusal, distinct-2); save as alpaca JSONL
  5. --filter pipes the JSONL through filter_dataset.py (dedup, teacher NLL)

Output is alpaca-schema JSONL + an HF dataset directory, both loadable by
load_dataset_split() for direct use as distillation training data.

Usage:
    # Quick bootstrap: generate 10 k pairs, keep best ~5 k
    python scripts/magpie_synth.py --n 10000 --output_dir ./magpie_data

    # With deep filter (dedup + teacher NLL re-ranking)
    python scripts/magpie_synth.py --n 20000 --output_dir ./magpie_data --filter --target 8000

    # Offline (uses cached teacher)
    python scripts/magpie_synth.py --n 5000 --output_dir ./magpie_data --offline
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from transformers import GenerationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import distinct2 as _distinct2, is_refusal, is_noise

# ── Fallback system prompts (used when domain registry is unavailable) ────────
SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are an expert software engineer. Help with programming and debugging.",
    "You are a mathematics tutor. Help solve problems step by step.",
    "You are a creative writing coach. Help with stories, poetry, and prose.",
    "You are a science educator. Explain concepts clearly with examples.",
    "You are a data analysis expert. Help with statistics and data interpretation.",
    "You are a language learning tutor. Help with grammar, vocabulary, and usage.",
    "You are a logic and reasoning coach. Help solve puzzles and structured problems.",
    "You are a writing editor. Help improve clarity, style, and argumentation.",
    "You are a curious generalist. Answer factual questions thoughtfully.",
    "You are a research assistant. Summarize and synthesize information.",
    "You are an expert in algorithms and computer science fundamentals.",
    "You are a philosophy discussion partner. Explore ideas rigorously.",
    "You are a practical life-skills advisor. Give concrete, actionable advice.",
    "You are a history and social science educator.",
    "You are a brainstorming partner. Generate and expand ideas creatively.",
]

_DEFAULT_FILTER_CFG = {
    "min_resp_words": 20, "max_resp_words": 600,
    "min_d2": 0.30, "require_code": False, "require_numbers": False,
}

# Default location of the domain registry JSON (relative to this file's parent)
_DEFAULT_DOMAINS_FILE = str(Path(__file__).parent.parent / "configs" / "domain_prompts.json")


def _load_domain_registry(domains_file: Path) -> tuple[dict, dict]:
    """Load domain system prompts and filter configs from a JSON registry.

    Returns (prompt_pools, filter_configs) where keys are domain names.
    Falls back gracefully if the file is missing or malformed.
    """
    if not domains_file.exists():
        LOG.warning("Domain registry not found: %s — using built-in general prompts.", domains_file)
        return {"general": SYSTEM_PROMPTS}, {"general": _DEFAULT_FILTER_CFG}

    try:
        with open(domains_file) as f:
            registry = json.load(f)

        prompt_pools: dict[str, list[str]] = {}
        filter_configs: dict[str, dict] = {}
        for key, val in registry.items():
            if key.startswith("_"):
                continue  # skip comment keys
            flt = val.get("filter", {})
            prompt_pools[key] = val.get("system_prompts", SYSTEM_PROMPTS)
            filter_configs[key] = {
                "min_resp_words":  flt.get("min_resp_words",  _DEFAULT_FILTER_CFG["min_resp_words"]),
                "max_resp_words":  flt.get("max_resp_words",  _DEFAULT_FILTER_CFG["max_resp_words"]),
                "min_d2":          flt.get("min_d2",          _DEFAULT_FILTER_CFG["min_d2"]),
                "require_code":    flt.get("require_code",    False),
                "require_numbers": flt.get("require_numbers", False),
            }
        LOG.info("Loaded %d domains from %s", len(prompt_pools), domains_file)
        return prompt_pools, filter_configs
    except (json.JSONDecodeError, KeyError, OSError) as e:
        LOG.error("Failed to load domain registry %s: %s — using built-in general prompts.", domains_file, e)
        return {"general": SYSTEM_PROMPTS}, {"general": _DEFAULT_FILTER_CFG}

# ── Inline quality filters (fast, no external deps) ─────────────────────────
MIN_INSTRUCTION_WORDS = 5
MAX_INSTRUCTION_WORDS = 180
MIN_RESPONSE_WORDS    = 20
MAX_RESPONSE_WORDS    = 600
MIN_DISTINCT2         = 0.30


def _passes_filter(instruction: str, response: str, filter_cfg: dict) -> bool:
    if not instruction or not response:
        return False
    iw = len(instruction.split())
    rw = len(response.split())
    if not (MIN_INSTRUCTION_WORDS <= iw <= MAX_INSTRUCTION_WORDS):
        return False
    if not (filter_cfg["min_resp_words"] <= rw <= filter_cfg["max_resp_words"]):
        return False
    if is_refusal(response):
        return False
    if is_noise(response) or is_noise(instruction):
        return False
    if _distinct2(response) < filter_cfg["min_d2"]:
        return False
    if filter_cfg.get("require_code") and not re.search(r"```|`[^`]+`", response):
        return False
    if filter_cfg.get("require_numbers") and not re.search(r"\d", response):
        return False
    return True


# ── Generation ───────────────────────────────────────────────────────────────

def _build_instruction_prefix(system_prompt: str) -> str:
    """Prefix that ends just before the user's message — model generates the instruction."""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"


def _build_response_prefix(system_prompt: str, instruction: str) -> str:
    """Prefix that ends just before the assistant's response."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _decode_new_tokens(tokenizer, output_ids, prefix_len: int, im_end_id: int) -> str:
    """Decode tokens generated after prefix_len; truncate at <|im_end|>."""
    new_ids = output_ids[prefix_len:]
    # Find first im_end in new tokens
    for i, tok in enumerate(new_ids):
        if tok == im_end_id:
            new_ids = new_ids[:i]
            break
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate_batch(model, tokenizer, prompts: list[str],
                   max_new_tokens: int, temperature: float,
                   device, im_end_id: int) -> list[str]:
    """Tokenize prompts (left-padded), generate, decode new tokens only."""
    import torch

    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=768,
    ).to(device)

    # Use the full padded input length as the cutoff — all sequences share the
    # same padded length so this correctly skips ALL input tokens (including pads).
    total_input_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=[tokenizer.eos_token_id, im_end_id],
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            ),
        )

    return [
        _decode_new_tokens(tokenizer, out[i].tolist(), total_input_len, im_end_id)
        for i in range(len(prompts))
    ]


def generate_batch_mlx(mlx_model, mlx_tokenizer, prompts: list[str],
                       max_new_tokens: int, temperature: float,
                       stop_str: str = "<|im_end|>") -> list[str]:
    """MLX-native generation — 2–4× faster than PyTorch MPS on Apple Silicon.

    mlx-lm does not support true batched autoregressive decode, but sequential
    generation with native Metal kernels is significantly faster than MPS.
    On M3 Max: ~80 tok/s for 1.5B vs ~20–30 tok/s for PyTorch MPS.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)
    results = []
    for prompt in prompts:
        out = generate(
            mlx_model, mlx_tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        # mlx-lm ≥0.10 returns only new tokens; older versions return prompt+new
        if out.startswith(prompt):
            out = out[len(prompt):]
        # Truncate at ChatML stop string
        if stop_str and stop_str in out:
            out = out[:out.index(stop_str)]
        results.append(out.strip())
    return results


# ── Convert JSONL → HF dataset ───────────────────────────────────────────────

def jsonl_to_hf_dataset(jsonl_path: Path, hf_dir: Path) -> int:
    from datasets import Dataset
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not items:
        return 0
    ds = Dataset.from_list(items)
    hf_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(hf_dir))
    return len(items)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Magpie self-synthesis for distillation")
    p.add_argument("--teacher", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                   help="Teacher model ID for generation (default: Qwen/Qwen2-1.5B-Instruct)")
    p.add_argument("--domain", type=str, default="general",
                   help="Domain key from the domain registry (default: general). "
                        "Must match a key in --domains_file. Custom domains saved via the UI are valid.")
    p.add_argument("--domains_file", type=str, default=_DEFAULT_DOMAINS_FILE,
                   help="Path to domain registry JSON (default: configs/domain_prompts.json). "
                        "Add new domains to this file to extend synthesis without code changes.")
    p.add_argument("--n", type=int, default=10000,
                   help="Number of pairs to GENERATE before filtering (default: 10000). "
                        "Expect ~40-60%% to survive inline filters.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write magpie_raw.jsonl + hf_dataset/")
    p.add_argument("--backend", type=str, default="auto",
                   choices=["auto", "mlx", "mps", "cuda", "cpu"],
                   help="Generation backend. 'auto' picks MLX on Apple Silicon if available, "
                        "else PyTorch (mps/cuda/cpu). MLX is 2–4× faster on M-series. (default: auto)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Generation batch size (default: 32; for MLX controls loop chunk size only — "
                        "MLX generates sequentially. Reduce for MPS/CUDA if OOM)")
    p.add_argument("--max_instruction_tokens", type=int, default=150,
                   help="Max new tokens for instruction generation (default: 150)")
    p.add_argument("--max_response_tokens", type=int, default=512,
                   help="Max new tokens for response generation (default: 512)")
    p.add_argument("--inst_temp", type=float, default=0.9,
                   help="Sampling temperature for instruction generation (default: 0.9)")
    p.add_argument("--resp_temp", type=float, default=0.7,
                   help="Sampling temperature for response generation (default: 0.7)")
    p.add_argument("--filter", action="store_true",
                   help="Run filter_dataset.py on the output (dedup + optional teacher NLL)")
    p.add_argument("--target", type=int, default=None,
                   help="Target size for --filter top-N selection (default: all that pass)")
    p.add_argument("--teacher_score", action="store_true",
                   help="Use teacher NLL scoring in filter step (slower, higher quality)")
    p.add_argument("--save_every", type=int, default=200,
                   help="Flush to disk every N kept examples (default: 200)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing magpie_raw.jsonl (skip already-generated count)")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true",
                   help="Air-gapped: use local HF cache only")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    import random as _random
    _random.seed(args.seed)

    import torch
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "magpie_raw.jsonl"
    hf_dir     = output_dir / "hf_dataset"

    offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    cache_dir = os.environ.get("HF_HOME") or args.cache_dir

    # ── Resolve backend ───────────────────────────────────────────────────────
    backend = args.backend
    if backend == "auto":
        try:
            import mlx.core  # noqa: F401
            from mlx_lm import load as _mlx_load_check  # noqa: F401
            backend = "mlx"
        except ImportError:
            if torch.backends.mps.is_available():
                backend = "mps"
            elif torch.cuda.is_available():
                backend = "cuda"
            else:
                backend = "cpu"
    LOG.info("Backend: %s", backend)

    # ── Load teacher ─────────────────────────────────────────────────────────
    LOG.info("Loading teacher: %s", args.teacher)

    if backend == "mlx":
        from mlx_lm import load as mlx_load
        mlx_model, mlx_tokenizer = mlx_load(args.teacher)
        # Extract underlying HF tokenizer for ChatML prefix building
        tokenizer = getattr(mlx_tokenizer, "_tokenizer", mlx_tokenizer)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        device = None
        im_end_id = None

        def _generate(prompts, max_new_tokens, temperature):
            return generate_batch_mlx(mlx_model, mlx_tokenizer, prompts,
                                      max_new_tokens, temperature)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if backend == "mps":
            device = torch.device("mps")
        elif backend == "cuda":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        tokenizer = AutoTokenizer.from_pretrained(
            args.teacher, cache_dir=cache_dir, local_files_only=offline,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            args.teacher,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=offline,
        )
        model.eval()

        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id == tokenizer.unk_token_id:
            im_end_id = tokenizer.eos_token_id
            LOG.warning("Model does not use <|im_end|> — stop token set to eos_token_id")

        def _generate(prompts, max_new_tokens, temperature):
            return generate_batch(model, tokenizer, prompts, max_new_tokens,
                                  temperature, device, im_end_id)

    # ── Load domain registry ──────────────────────────────────────────────────
    prompt_pools, filter_configs = _load_domain_registry(Path(args.domains_file))

    domain = args.domain
    if domain not in prompt_pools:
        available = sorted(prompt_pools.keys())
        LOG.warning("Domain '%s' not found in registry. Available: %s. Falling back to 'general'.",
                    domain, available)
        domain = "general" if "general" in prompt_pools else available[0]

    prompt_pool = prompt_pools[domain]
    filter_cfg  = filter_configs.get(domain, _DEFAULT_FILTER_CFG)
    LOG.info("Domain: %s  |  system prompt pool size: %d", domain, len(prompt_pool))

    # ── Resume: count already-generated examples ─────────────────────────────
    n_already = 0
    if args.resume and jsonl_path.exists():
        with open(jsonl_path) as f:
            n_already = sum(1 for line in f if line.strip())
        LOG.info("Resuming: found %d existing examples, generating %d more",
                 n_already, max(0, args.n - n_already))
    n_to_generate = max(0, args.n - n_already)

    if n_to_generate == 0:
        LOG.info("Already have %d examples (--n %d). Skipping generation.", n_already, args.n)
    else:
        # ── Generation loop ──────────────────────────────────────────────────
        n_generated = 0
        n_kept      = n_already
        stats = {"empty_instr": 0, "empty_resp": 0, "filtered": 0}

        with open(jsonl_path, "a", buffering=1) as fout:
            sys_cycle = 0
            while n_generated < n_to_generate:
                batch = min(args.batch_size, n_to_generate - n_generated)

                # Pick system prompts (cycle through domain pool for diversity)
                sys_prompts = [
                    prompt_pool[(sys_cycle + i) % len(prompt_pool)]
                    for i in range(batch)
                ]
                sys_cycle = (sys_cycle + batch) % len(prompt_pool)

                # ── Phase 1: generate instructions ───────────────────────────
                inst_prefixes = [_build_instruction_prefix(sp) for sp in sys_prompts]
                try:
                    instructions = _generate(
                        inst_prefixes,
                        max_new_tokens=args.max_instruction_tokens,
                        temperature=args.inst_temp,
                    )
                except Exception as e:
                    LOG.warning("Instruction batch failed: %s", e)
                    n_generated += batch
                    continue

                # Drop empty instructions
                valid = [(sp, instr) for sp, instr in zip(sys_prompts, instructions) if instr]
                stats["empty_instr"] += batch - len(valid)

                if not valid:
                    n_generated += batch
                    continue

                valid_sys, valid_instr = zip(*valid)

                # ── Phase 2: generate responses ──────────────────────────────
                resp_prefixes = [
                    _build_response_prefix(sp, instr)
                    for sp, instr in zip(valid_sys, valid_instr)
                ]
                try:
                    responses = _generate(
                        resp_prefixes,
                        max_new_tokens=args.max_response_tokens,
                        temperature=args.resp_temp,
                    )
                except Exception as e:
                    LOG.warning("Response batch failed: %s", e)
                    n_generated += batch
                    continue

                # ── Inline filter + save ─────────────────────────────────────
                for instr, resp in zip(valid_instr, responses):
                    if not resp:
                        stats["empty_resp"] += 1
                        continue
                    if not _passes_filter(instr, resp, filter_cfg):
                        stats["filtered"] += 1
                        continue
                    fout.write(json.dumps({"instruction": instr, "input": "", "output": resp, "domain": domain}))
                    fout.write("\n")
                    n_kept += 1

                n_generated += batch

                if n_generated % args.save_every == 0 or n_generated >= n_to_generate:
                    pass_rate = (n_kept - n_already) / max(n_generated, 1) * 100
                    LOG.info(
                        "Progress: generated=%d/%d  kept=%d  pass_rate=%.1f%%  "
                        "(empty_instr=%d empty_resp=%d filtered=%d)",
                        n_generated, n_to_generate, n_kept, pass_rate,
                        stats["empty_instr"], stats["empty_resp"], stats["filtered"],
                    )
                    # Flush JSONL (crash-safe artifact); HF dataset rebuilt once at end.
                    fout.flush()

        LOG.info("Generation complete: %d generated, %d kept (%.1f%% pass rate)",
                 n_generated, n_kept - n_already,
                 (n_kept - n_already) / max(n_generated, 1) * 100)

    # ── Build HF dataset from JSONL ──────────────────────────────────────────
    n_total = jsonl_to_hf_dataset(jsonl_path, hf_dir)
    LOG.info("HF dataset saved: %s  (%d examples)", hf_dir, n_total)

    # ── Optional: deep filter ────────────────────────────────────────────────
    if args.filter:
        filtered_dir = output_dir / "filtered"
        filter_cmd = [
            sys.executable, "scripts/filter_dataset.py",
            "--dataset", str(hf_dir),
            "--output_dir", str(filtered_dir),
            "--minhash",
        ]
        if args.target:
            filter_cmd += ["--target", str(args.target)]
        if args.teacher_score:
            filter_cmd += ["--teacher_score", "--teacher", args.teacher]
        if args.offline:
            filter_cmd += ["--offline"]

        LOG.info("Running filter_dataset.py → %s", filtered_dir)
        import subprocess
        result = subprocess.run(filter_cmd, cwd=Path(__file__).parent.parent)
        if result.returncode == 0:
            LOG.info("Filtered dataset saved: %s", filtered_dir)
        else:
            LOG.warning("filter_dataset.py exited with code %d", result.returncode)

    LOG.info(
        "Done [domain=%s]. Use with distillation:\n"
        "  python scripts/run_distillation_agent.py --dataset %s --backend mlx --open",
        domain,
        hf_dir if not args.filter else output_dir / "filtered",
    )


if __name__ == "__main__":
    main()
