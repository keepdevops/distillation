#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.run_eval"""
import importlib

_m = importlib.import_module("distill.run_eval")
if __name__ == "__main__":
    getattr(_m, "main")()
