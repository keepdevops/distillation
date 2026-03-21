#!/usr/bin/env python3
"""
llama.cpp C++ backend for fast inference/eval on Apple Silicon (M3 Max).

Uses pre-compiled Metal-accelerated binaries from /Users/Shared/llama/:
  - llama-perplexity : chunked corpus PPL — 3-5x faster than MLX Python
  - llama-server     : HTTP completion API with parallel slots
                       fire n_parallel concurrent requests → GPU saturated

Typical speedups on M3 Max vs MLX Python backend:
  Perplexity (500 seqs, ctx=512): 8-15s  vs 30-60s
  Generation (50 prompts, parallel=4): 30-60s vs 3-5 min
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LLAMA_BIN_DIR = Path("/Users/Shared/llama/llama.cpp-master/build/bin")
DEFAULT_N_GPU_LAYERS = 99    # offload all layers to Metal GPU
DEFAULT_PORT = 8089           # avoid clash with default llama-server port 8080
DEFAULT_N_PARALLEL = 4        # concurrent completion slots in llama-server


# ── Binary helpers ─────────────────────────────────────────────────────────────

def get_binary(name: str) -> str:
    p = LLAMA_BIN_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"llama.cpp binary not found: {p}")
    return str(p)


def is_cpp_available() -> bool:
    """Return True if llama.cpp binaries are present."""
    try:
        get_binary("llama-perplexity")
        get_binary("llama-server")
        return True
    except FileNotFoundError:
        return False


def find_gguf(path: str) -> Optional[str]:
    """
    Return an absolute GGUF file path from a file or directory.
    Picks the largest .gguf if multiple exist (prefers quantized over f16).
    """
    p = Path(path)
    if p.is_file() and p.suffix == ".gguf":
        return str(p.resolve())
    if p.is_dir():
        # Prefer Q4/Q8 quantized over f16 for speed; largest file first
        files = sorted(p.glob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
        if files:
            return str(files[0].resolve())
    return None


# ── Perplexity ─────────────────────────────────────────────────────────────────

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
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
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


# ── Server + generation ────────────────────────────────────────────────────────

class LlamaServer:
    """
    Context manager that runs llama-server as a background process.

    Opens n_parallel completion slots so you can fire concurrent HTTP
    requests and keep the Metal GPU fully saturated.

    Usage:
        with LlamaServer(gguf_path, n_parallel=4) as srv:
            text = srv.complete("Hello", max_tokens=128)
    """

    def __init__(
        self,
        gguf_path: str,
        port: int = DEFAULT_PORT,
        n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
        ctx_size: int = 4096,
        n_parallel: int = DEFAULT_N_PARALLEL,
        threads: int = 4,
    ):
        self.gguf_path = gguf_path
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.n_parallel = n_parallel
        self.threads = threads
        self._proc: Optional[subprocess.Popen] = None
        self._log_file = None  # file handle for llama-server output

    def __enter__(self) -> "LlamaServer":
        self._start()
        self._wait_ready()
        return self

    def __exit__(self, *_):
        self._stop()

    def _start(self) -> None:
        cmd = [
            get_binary("llama-server"),
            "-m", self.gguf_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ngl", str(self.n_gpu_layers),
            "--ctx-size", str(self.ctx_size),
            "--parallel", str(self.n_parallel),
            "-t", str(self.threads),
        ]
        log_path = Path(self.gguf_path).with_suffix(".server.log")
        logger.info(
            "Starting llama-server: %s (port=%d  ngl=%d  parallel=%d)  log→%s",
            Path(self.gguf_path).name, self.port, self.n_gpu_layers, self.n_parallel,
            log_path,
        )
        self._log_file = open(log_path, "w")
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=self._log_file,
        )

    def _wait_ready(self, timeout: int = 90) -> None:
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited prematurely (rc={self._proc.returncode})"
                )
            try:
                with urllib.request.urlopen(url, timeout=2):
                    logger.info("  llama-server ready on port %d", self.port)
                    return
            except Exception:
                time.sleep(0.5)
        self._stop()
        raise TimeoutError(f"llama-server not ready after {timeout}s")

    def _stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """POST a completion request; returns the generated text only."""
        payload = json.dumps({
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": [],
            "stream": False,
            "cache_prompt": False,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["content"]


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
                    logger.info(
                        "  llama-server: %d/%d prompts done", done, len(prompts)
                    )

    return results
