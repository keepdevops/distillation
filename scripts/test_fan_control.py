#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.test_fan_control"""
import importlib

_m = importlib.import_module("distill.test_fan_control")
if __name__ == "__main__":
    getattr(_m, "main")()
