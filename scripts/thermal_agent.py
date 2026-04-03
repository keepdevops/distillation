#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.thermal_agent"""
import importlib

_m = importlib.import_module("distill.thermal_agent")
if __name__ == "__main__":
    getattr(_m, "main")()
