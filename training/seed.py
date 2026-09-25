import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """
    Sets the random seed across Python random, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed (int): The seed value to set.
        deterministic (bool): If True, configures PyTorch CUDNN backends for determinism.

    Returns:
        int: The set seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
