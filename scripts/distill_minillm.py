#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.distill_minillm"""
import importlib

_m = importlib.import_module("distill.distill_minillm")
if __name__ == "__main__":
    getattr(_m, "main")()
