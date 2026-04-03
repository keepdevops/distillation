# compatibility shim — module has moved to distill.ui.plot_training
from distill.ui.plot_training import *  # noqa: F401, F403
from distill.ui.plot_training import main  # noqa: F401

if __name__ == "__main__":
    main()
