"""Compatibility alias — the canonical package name is ``distill`` (two l's).

Exists so that ``python -m distil.<module>`` resolves correctly even when
the module is invoked with one 'l' by mistake.
"""
from distill import *  # noqa: F401, F403
