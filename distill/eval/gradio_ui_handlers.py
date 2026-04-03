"""
Event-handler factories and helpers for the Universal Gradio UI.

Separates closure-building logic from the Gradio layout so that
gradio_ui.py stays focused on orchestration.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_artifact_info(artifacts_info, path: str) -> str:
    """Return a Markdown string summarising detected artifacts.

    Args:
        artifacts_info: Dict returned by ``detect_artifacts``, or ``None``
                        when ``path`` points to a single file.
        path: String path that was loaded (used for display only).

    Returns:
        Markdown-formatted summary string.
    """
    if artifacts_info is None:
        return "Not a directory - single file loaded"

    lines = [
        f"**Path:** `{Path(path).name}`",
        f"**Training method:** {artifacts_info['training_method']}",
        f"**Available formats:** {', '.join(artifacts_info['formats']) if artifacts_info['formats'] else 'None'}",
    ]

    if artifacts_info["has_metrics"]:
        lines.append("**Metrics:** \u2705 Available")
    else:
        lines.append("**Metrics:** \u274c Not found")

    if artifacts_info["checkpoints"]:
        ckpts = ", ".join([f"step {c['step']}" for c in artifacts_info["checkpoints"][:5]])
        if len(artifacts_info["checkpoints"]) > 5:
            ckpts += f" (+{len(artifacts_info['checkpoints']) - 5} more)"
        lines.append(f"**Checkpoints:** {ckpts}")

    if artifacts_info["artifacts"]:
        lines.append("\n**Artifacts:**")
        for name, typ, _, size_gb in artifacts_info["artifacts"]:
            lines.append(f"- `{name}` ({typ}, {size_gb:.2f} GB)")

    return "\n".join(lines)


def make_load_model_fn(loader, model_loaded: dict, path: str, backend: str):
    """Factory that returns a ``load_model_fn`` closure bound to *loader*.

    Args:
        loader: ``UniversalModelLoader`` instance.
        model_loaded: Mutable dict with keys ``"loaded"`` and ``"message"``.
        path: Resolved model path string.
        backend: Default backend string used when caller passes no override.

    Returns:
        Callable suitable for wiring to a Gradio ``Button.click`` event.
    """
    import gradio as gr

    def load_model_fn(selected_backend: str):
        """Load model with selected backend."""
        nonlocal model_loaded

        effective_backend = selected_backend if selected_backend else backend
        logger.info("Loading model with backend: %s", effective_backend)

        try:
            success, message = loader.load(path, backend=effective_backend)
        except Exception as exc:
            logger.error("Unexpected error while loading model: %s", exc, exc_info=True)
            model_loaded["loaded"] = False
            model_loaded["message"] = str(exc)
            return f"\u274c Unexpected error: {exc}", gr.update(interactive=False)

        if success:
            info = loader.get_info()
            model_loaded["loaded"] = True
            model_loaded["message"] = message
            status_msg = f"\u2705 {message}\n\nBackend: {info['backend']}"
            if "device" in info:
                status_msg += f"\nDevice: {info['device']}"
            if "dtype" in info:
                status_msg += f"\nDtype: {info['dtype']}"
            return status_msg, gr.update(interactive=True)
        else:
            model_loaded["loaded"] = False
            model_loaded["message"] = message
            logger.error("Model load failed: %s", message)
            return f"\u274c {message}", gr.update(interactive=False)

    return load_model_fn


def make_generate_fn(loader, model_loaded: dict):
    """Factory that returns a ``generate_fn`` closure bound to *loader*.

    Args:
        loader: ``UniversalModelLoader`` instance (must already be loaded).
        model_loaded: Mutable dict with key ``"loaded"`` (bool).

    Returns:
        Callable suitable for wiring to a Gradio ``Button.click`` event.
    """

    def generate_fn(prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text from prompt."""
        if not model_loaded["loaded"]:
            return "\u26a0\ufe0f Please load a model first"

        if not prompt.strip():
            return ""

        logger.info("Generating (max_tokens=%d, temp=%.2f)", int(max_tokens), temperature)
        try:
            result = loader.generate(
                prompt,
                max_new_tokens=int(max_tokens),
                temperature=temperature,
            )
        except Exception as exc:
            logger.error("Generation error: %s", exc, exc_info=True)
            return f"\u274c Generation error: {exc}"
        return result

    return generate_fn
