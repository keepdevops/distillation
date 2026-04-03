# compatibility shim — module has moved to distill.data.synth
from distill.data.synth import *  # noqa: F401, F403
from distill.data.synth import main  # noqa: F401

if __name__ == "__main__":
    main()
