# compatibility shim — module has moved to distill.ui.dashboard
from distill.ui.dashboard import *  # noqa: F401, F403
from distill.ui.dashboard import main  # noqa: F401

if __name__ == "__main__":
    main()
