"""distill.cpp — optional compiled extension for high-performance telemetry.

Importing this package succeeds even when the C++ module has not been compiled.
Use ``distill.cpp.is_available()`` to check at runtime.
"""
from __future__ import annotations


def is_available() -> bool:
    """Return True if the distill_cpp native extension is compiled and loadable."""
    try:
        import importlib
        importlib.import_module("distill_cpp")
        return True
    except ImportError:
        return False


def get_module():
    """Return the distill_cpp module, or None if not available."""
    try:
        import importlib
        return importlib.import_module("distill_cpp")
    except ImportError:
        return None
