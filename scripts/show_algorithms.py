#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.show_algorithms"""
import importlib

_m = importlib.import_module("distill.show_algorithms")
if __name__ == "__main__":
    getattr(_m, "main")()
