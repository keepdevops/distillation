# compatibility shim — module has moved to distill.eval.gradio_ui
from distill.eval.gradio_ui import *  # noqa: F401, F403
from distill.eval.gradio_ui import main  # noqa: F401

if __name__ == "__main__":
    main()
