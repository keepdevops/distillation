# compatibility shim — module has moved to distill.training.backends.unsloth
from distill.training.backends.unsloth import *  # noqa: F401, F403
from distill.training.backends.unsloth import main  # noqa: F401

if __name__ == "__main__":
    main()
