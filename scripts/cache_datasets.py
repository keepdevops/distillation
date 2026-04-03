#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.cache_datasets"""
import importlib

_m = importlib.import_module("distill.cache_datasets")
if __name__ == "__main__":
    getattr(_m, "main")()
