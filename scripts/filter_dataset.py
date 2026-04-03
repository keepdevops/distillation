#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.filter_dataset"""
import importlib

_m = importlib.import_module("distill.filter_dataset")
if __name__ == "__main__":
    getattr(_m, "main")()
