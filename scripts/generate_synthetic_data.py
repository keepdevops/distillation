#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.generate_synthetic_data"""
import importlib

_m = importlib.import_module("distill.generate_synthetic_data")
if __name__ == "__main__":
    getattr(_m, "main")()
