#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.dashboard"""
import importlib

_m = importlib.import_module("distill.dashboard")
if __name__ == "__main__":
    getattr(_m, "main")()
