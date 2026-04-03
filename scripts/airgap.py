#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.airgap"""
import importlib

_m = importlib.import_module("distill.airgap")
if __name__ == "__main__":
    getattr(_m, "main")()
