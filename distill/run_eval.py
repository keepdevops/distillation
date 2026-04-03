# compatibility shim — module has moved to distill.eval.perplexity
from distill.eval.perplexity import *  # noqa: F401, F403
from distill.eval.perplexity import main  # noqa: F401

if __name__ == "__main__":
    main()
