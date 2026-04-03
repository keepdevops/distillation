#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.distill_mlx"""
import importlib

_m = importlib.import_module("distill.distill_mlx")
if __name__ == "__main__":
    getattr(_m, "main")()
