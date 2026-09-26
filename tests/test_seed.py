import torch
import numpy as np
import random
from training.seed import set_seed

def test_set_seed_reproducibility():
    set_seed(1234)
    py_val_1 = random.randint(0, 100000)
    np_val_1 = np.random.rand(5)
    torch_val_1 = torch.randn(3, 3)

    set_seed(1234)
    py_val_2 = random.randint(0, 100000)
    np_val_2 = np.random.rand(5)
    torch_val_2 = torch.randn(3, 3)

    assert py_val_1 == py_val_2
    assert np.allclose(np_val_1, np_val_2)
    assert torch.equal(torch_val_1, torch_val_2)

def test_set_seed_options_and_reconfiguration():
    # Test setting deterministic=True and use_deterministic_algorithms=True
    seed1 = set_seed(999, deterministic=True, benchmark=True, use_deterministic_algorithms=True)
    assert seed1 == 999
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.deterministic is True
        # benchmark should be forced to False when deterministic=True
        assert torch.backends.cudnn.benchmark is False

    # Test reconfiguring use_deterministic_algorithms=False resets state
    seed2 = set_seed(888, deterministic=False, benchmark=True, use_deterministic_algorithms=False)
    assert seed2 == 888
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
