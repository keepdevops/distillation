#!/usr/bin/env python3
"""Backward-compatible shim. Prefer: python -m distill.monitor_cpu_gpu_temp"""
import importlib

_m = importlib.import_module("distill.monitor_cpu_gpu_temp")
if __name__ == "__main__":
    getattr(_m, "main")()
