"""Compatibility shim — redirects ``python -m distil.eval_gradio`` to ``distill.eval_gradio``."""
from distill.eval_gradio import *  # noqa: F401, F403
from distill.eval_gradio import main  # noqa: F401

if __name__ == "__main__":
    main()
