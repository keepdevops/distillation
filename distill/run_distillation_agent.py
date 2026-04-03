# compatibility shim — module has moved to distill.orchestration.agent
from distill.orchestration.agent import *  # noqa: F401, F403
from distill.orchestration.agent import main  # noqa: F401

if __name__ == "__main__":
    main()
