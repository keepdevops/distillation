#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.plot_training"""
import importlib

_m = importlib.import_module("distill.plot_training")
if __name__ == "__main__":
    getattr(_m, "main")()
