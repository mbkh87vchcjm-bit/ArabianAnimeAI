import random
import numpy as np
import torch

def set_seed(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False,
    use_deterministic_algorithms: bool = False
) -> int:
    """
    Sets random seeds across Python random, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed (int): The seed value to set.
        deterministic (bool): If True, configures PyTorch CUDNN backends for determinism (torch.backends.cudnn.deterministic).
        benchmark (bool): If True, enables PyTorch CUDNN benchmarking for performance (torch.backends.cudnn.benchmark).
        use_deterministic_algorithms (bool): If True, configures PyTorch to throw errors for non-deterministic operations.

    Note:
        Full mathematical reproducibility across hardware, CUDA driver versions, or PyTorch operators is NOT
        guaranteed solely by setting a seed. Certain CUDA operations or custom kernels may exhibit minor non-determinism.

    Returns:
        int: The set seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

    if use_deterministic_algorithms and hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Fallback if specific hardware operations do not support strict deterministic mode
            pass

    return seed
