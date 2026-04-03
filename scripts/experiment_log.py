#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.experiment_log"""
import importlib

_m = importlib.import_module("distill.experiment_log")
if __name__ == "__main__":
    getattr(_m, "main")()
