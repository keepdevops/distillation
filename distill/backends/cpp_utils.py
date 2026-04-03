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
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LLAMA_ROOT = Path("/Users/Shared/llama")
DEFAULT_N_GPU_LAYERS = 99    # offload all layers to Metal GPU
DEFAULT_PORT = 8089           # avoid clash with default llama-server port 8080
DEFAULT_N_PARALLEL = 4        # concurrent completion slots in llama-server

# Candidate locations searched in order for each binary
_LLAMA_BIN_CANDIDATES = [
    LLAMA_ROOT,                                        # flat: /Users/Shared/llama/llama-server
    LLAMA_ROOT / "llama.cpp" / "build" / "bin",       # cmake build
    LLAMA_ROOT / "llama.cpp-master" / "build" / "bin",
    LLAMA_ROOT / "build" / "bin",
]


# ── Binary helpers ─────────────────────────────────────────────────────────────

def get_binary(name: str) -> str:
    for d in _LLAMA_BIN_CANDIDATES:
        p = d / name
        if p.exists():
            return str(p)
    searched = ", ".join(str(d / name) for d in _LLAMA_BIN_CANDIDATES)
    raise FileNotFoundError(f"llama.cpp binary '{name}' not found. Searched: {searched}")


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
            "-ngl", str(self.n_gpu_layers),
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
                tail = ""
                if self._log_file:
                    self._log_file.flush()
                    log_path = Path(self._log_file.name)
                    try:
                        lines = log_path.read_text().splitlines()
                        tail = "\n".join(lines[-20:])
                    except Exception:
                        pass
                raise RuntimeError(
                    f"llama-server exited prematurely (rc={self._proc.returncode})\n"
                    f"Last log lines:\n{tail}"
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
        timeout: int | None = None,
    ) -> str:
        """POST a completion request; returns the generated text only.

        timeout defaults to max_tokens * 0.6 + 90 (generous for large models /
        parallel slots on M3 Max at ~15 tok/s worst-case).
        """
        if timeout is None:
            timeout = max(120, int(max_tokens * 0.6) + 90)
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["content"]


# Inference functions live in cpp_inference.py to keep this file under 300 LOC.
# Re-export for backward compatibility of any callers importing from here.
# (Import is deferred to avoid circular imports at module load time.)
def compute_gguf_perplexity(*args, **kwargs):  # noqa: F401
    from .cpp_inference import compute_gguf_perplexity as _fn
    return _fn(*args, **kwargs)


def generate_gguf_responses(*args, **kwargs):  # noqa: F401
    from .cpp_inference import generate_gguf_responses as _fn
    return _fn(*args, **kwargs)
