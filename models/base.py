from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Dict

class BaseModel(nn.Module, ABC):
    """
    Abstract Base Class for all models within ArabianAnimeAI (e.g. Video VAE, Generation Transformer).
    """

    def __init__(self, config: Any = None):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    def get_num_params(self) -> int:
        """Returns total trainable parameters count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
