#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.run_benchmarks"""
import importlib

_m = importlib.import_module("distill.run_benchmarks")
if __name__ == "__main__":
    getattr(_m, "main")()
