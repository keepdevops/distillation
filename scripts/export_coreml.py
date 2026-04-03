#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.export_coreml"""
import importlib

_m = importlib.import_module("distill.export_coreml")
if __name__ == "__main__":
    getattr(_m, "main")()
