# compatibility shim — module has moved to distill.monitoring.thermal
from distill.monitoring.thermal import *  # noqa: F401, F403
from distill.monitoring.thermal import main  # noqa: F401

if __name__ == "__main__":
    main()
