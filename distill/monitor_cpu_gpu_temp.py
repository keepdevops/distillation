# compatibility shim — module has moved to distill.monitoring.temp_logger
from distill.monitoring.temp_logger import *  # noqa: F401, F403
from distill.monitoring.temp_logger import main  # noqa: F401

if __name__ == "__main__":
    main()
