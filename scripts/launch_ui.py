#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.launch_ui.main"""
import importlib

_m = importlib.import_module("distill.launch_ui.main")
if __name__ == "__main__":
    getattr(_m, "main")()
