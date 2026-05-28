# compatibility shim ? module has moved to distill.training.backends.minillm
from distill.training.backends.minillm import *  # noqa: F401, F403
from distill.training.backends.minillm import main  # noqa: F401

if __name__ == "__main__":
    main()
