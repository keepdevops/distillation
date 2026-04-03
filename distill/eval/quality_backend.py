"""
Backend selection and student model loading for quality eval.

Provides:
    select_backend(args, checkpoint_dir) -> (use_gguf, use_mlx, gguf_student_path)
    load_student_backend(use_gguf, use_mlx, checkpoint_dir, args, cache_dir, offline, device) -> (model, tokenizer)
"""

import logging
from pathlib import Path

from ..backends.mlx_utils import is_mlx_available, load_mlx_model
from ..backends.cpp_utils import is_cpp_available, find_gguf
from ..infra.train_utils import load_student_model

logger = logging.getLogger(__name__)


def select_backend(args, checkpoint_dir: Path) -> tuple:
    """Determine which backend to use based on args and environment.

    Returns:
        (use_gguf, use_mlx, gguf_student_path)
    """
    use_gguf = False
    use_mlx = False
    gguf_student_path = None

    if args.backend == "gguf":
        if not is_cpp_available():
            logger.warning(
                "GGUF backend requested but llama.cpp binaries not found, falling back"
            )
        else:
            use_gguf = True
            logger.info("Backend: GGUF/llama.cpp (Metal, parallel generation)")

    elif args.backend == "mlx":
        if not is_mlx_available():
            logger.warning(
                "MLX backend requested but mlx/mlx-lm not installed, falling back to PyTorch"
            )
        else:
            use_mlx = True
            logger.info("Backend: MLX (Apple Silicon optimized)")

    elif args.backend == "auto":
        gguf_candidate = find_gguf(str(checkpoint_dir))
        if gguf_candidate and is_cpp_available():
            use_gguf = True
            logger.info(
                "Backend: GGUF/llama.cpp (auto-detected: %s)",
                Path(gguf_candidate).name,
            )
        elif is_mlx_available():
            use_mlx = True
            logger.info("Backend: MLX (auto-detected)")
        else:
            logger.info("Backend: PyTorch, Device: %s", args.backend)
    else:
        logger.info("Backend: PyTorch (explicit)")

    # Resolve GGUF student path
    if use_gguf:
        gguf_student_path = find_gguf(str(checkpoint_dir))
        if gguf_student_path is None:
            logger.warning(
                "No .gguf found in %s — falling back to MLX/PyTorch", checkpoint_dir
            )
            use_gguf = False
            use_mlx = is_mlx_available()

    return use_gguf, use_mlx, gguf_student_path


def load_student_backend(
    use_gguf: bool,
    use_mlx: bool,
    checkpoint_dir: Path,
    args,
    cache_dir: str,
    offline: bool,
    device,
) -> tuple:
    """Load student model and tokenizer for the chosen backend.

    Returns:
        (student, tokenizer) — both None for GGUF (generation via subprocess)
    """
    logger.info("Loading student from %s", checkpoint_dir)

    if use_gguf:
        logger.info(
            "GGUF student loaded via llama-server (model/tokenizer not loaded in-process)"
        )
        return None, None

    if use_mlx:
        student, tokenizer = load_mlx_model(checkpoint_dir, student_name=args.student)
        return student, tokenizer

    student, tokenizer = load_student_model(
        checkpoint_dir, args.student, cache_dir, offline, device
    )
    return student, tokenizer
