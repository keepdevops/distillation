# compatibility shim — module has moved to distill.monitoring.watchdog
from distill.monitoring.watchdog import *  # noqa: F401, F403
from distill.monitoring.watchdog import main  # noqa: F401

if __name__ == "__main__":
    main()
