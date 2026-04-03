#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.fan_control_popup"""
import importlib

_m = importlib.import_module("distill.fan_control_popup")
if __name__ == "__main__":
    getattr(_m, "main")()
