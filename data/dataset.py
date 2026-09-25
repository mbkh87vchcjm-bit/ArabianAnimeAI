from abc import ABC, abstractmethod
from typing import Any, Dict
from torch.utils.data import Dataset as PyTorchDataset

class BaseDataset(PyTorchDataset, ABC):
    """
    Abstract Base Class for dataset representations in ArabianAnimeAI.
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        pass
