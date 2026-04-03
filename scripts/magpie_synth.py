#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.magpie_synth"""
import importlib

_m = importlib.import_module("distill.magpie_synth")
if __name__ == "__main__":
    getattr(_m, "main")()
