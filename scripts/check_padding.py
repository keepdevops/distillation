#!/usr/bin/env python3
"""Shim — prefer: python -m distill.check_padding"""
import runpy

runpy.run_module("distill.check_padding", run_name="__main__")
