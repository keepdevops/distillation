# compatibility shim — module has moved to distill.monitoring.test_fan_control
from distill.monitoring.test_fan_control import *  # noqa: F401, F403
from distill.monitoring.test_fan_control import main  # noqa: F401

if __name__ == "__main__":
    main()
