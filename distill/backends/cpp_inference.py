"""GGUF inference functions: perplexity and parallel response generation."""
from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .cpp_utils import (
    get_binary, DEFAULT_N_GPU_LAYERS, DEFAULT_PORT, DEFAULT_N_PARALLEL, LlamaServer,
)

logger = logging.getLogger(__name__)


def compute_gguf_perplexity(
    gguf_path: str,
    texts: list[str],
    ctx_size: int = 512,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    threads: int = 4,
) -> Optional[float]:
    """
    Compute corpus perplexity using llama-perplexity (C++ / Metal).

    Concatenates all texts into a single temp file, runs llama-perplexity
    with full Metal GPU offload (-ngl 99), parses the final PPL estimate,
    and returns the mean NLL (natural log) for direct comparison with
    compute_mlx_perplexity / eval_loss outputs.

    Note: llama-perplexity uses a sliding-window over the concatenated
    corpus, which gives slightly different numbers than per-sequence eval.
    Both metrics track the same signal; use one consistently.
    """
    combined = "\n\n".join(t.strip() for t in texts if t.strip())
    if not combined:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(combined)
        tmp_path = f.name

    try:
        cmd = [
            get_binary("llama-perplexity"),
            "-m", gguf_path,
            "-f", tmp_path,
            "--ctx-size", str(ctx_size),
            "--ubatch-size", str(ctx_size),
            "--ngl", str(n_gpu_layers),
            "-t", str(threads),
        ]
        logger.info(
            "llama-perplexity: %s  ctx=%d  ngl=%d",
            Path(gguf_path).name, ctx_size, n_gpu_layers,
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr

        m = re.search(r"Final estimate:\s+PPL\s*=\s*([\d.]+)", output)
        if m:
            ppl = float(m.group(1))
            nll = math.log(ppl)
            logger.info("  PPL = %.4f  (NLL = %.4f)", ppl, nll)
            return nll

        logger.error(
            "Could not parse PPL output (exit=%d):\n%s",
            result.returncode, output[-1000:],
        )
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def generate_gguf_responses(
    gguf_path: str,
    prompts: list[str],
    max_tokens: int = 512,
    temperature: float = 0.7,
    port: int = DEFAULT_PORT,
    n_parallel: int = DEFAULT_N_PARALLEL,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    ctx_size: int = 4096,
    threads: int = 4,
) -> list[str]:
    """
    Generate responses for all prompts using llama-server with parallel slots.

    Starts one server process, then fires up to n_parallel concurrent HTTP
    requests via ThreadPoolExecutor — Metal GPU is kept saturated across all
    slots the entire time. Typically 3-8× faster than sequential Python
    generation for batches of 20+ prompts.
    """
    results: list[str] = [""] * len(prompts)

    with LlamaServer(
        gguf_path,
        port=port,
        n_gpu_layers=n_gpu_layers,
        ctx_size=ctx_size,
        n_parallel=n_parallel,
        threads=threads,
    ) as srv:
        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            futures = {
                pool.submit(srv.complete, p, max_tokens, temperature): i
                for i, p in enumerate(prompts)
            }
            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    logger.warning("Prompt %d generation failed: %s", idx, e)
                done += 1
                if done % 10 == 0:
                    logger.info("  llama-server: %d/%d prompts done", done, len(prompts))

    return results
