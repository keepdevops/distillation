# compatibility shim — module has moved to distill.export.gguf_pipeline
from distill.export.gguf_pipeline import *  # noqa: F401, F403
from distill.export.gguf_pipeline import main  # noqa: F401

if __name__ == "__main__":
    main()
