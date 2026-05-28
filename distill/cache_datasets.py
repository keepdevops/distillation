# compatibility shim — module has moved to distill.orchestration.cache_datasets
from distill.orchestration.cache_datasets import *  # noqa: F401, F403
from distill.orchestration.cache_datasets import main  # noqa: F401

if __name__ == "__main__":
    main()
