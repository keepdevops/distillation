#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.eval_quality"""
import importlib

_m = importlib.import_module("distill.eval_quality")
if __name__ == "__main__":
    getattr(_m, "main")()
