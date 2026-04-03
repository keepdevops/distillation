#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.training_watchdog"""
import importlib

_m = importlib.import_module("distill.training_watchdog")
if __name__ == "__main__":
    getattr(_m, "main")()
