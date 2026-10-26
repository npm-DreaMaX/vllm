"""Use the official CLI with an explicit NumPy seed for synthetic requests."""
import numpy as np

from vllm.entrypoints.cli.main import main

if __name__ == '__main__':
    np.random.seed(42)
    main()
