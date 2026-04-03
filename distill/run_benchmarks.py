# compatibility shim — module has moved to distill.eval.benchmarks
from distill.eval.benchmarks import *  # noqa: F401, F403
from distill.eval.benchmarks import main  # noqa: F401

if __name__ == "__main__":
    main()
