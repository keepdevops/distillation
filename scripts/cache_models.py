#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.cache_models"""
import importlib

_m = importlib.import_module("distill.cache_models")
if __name__ == "__main__":
    getattr(_m, "main")()
