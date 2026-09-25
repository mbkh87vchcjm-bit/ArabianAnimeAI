from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

class BaseTextConditioner(nn.Module, ABC):
    """
    Abstract Interface for Story/Prompt Compiler and Text Embedder modules.
    Translates raw text prompts and scene descriptions into latent condition embeddings.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, prompts: list[str], **kwargs: Any) -> torch.Tensor:
        """
        Args:
            prompts (list[str]): List of prompt strings.
        Returns:
            torch.Tensor: Text embeddings tensor of shape (batch_size, seq_len, embed_dim).
        """
        pass
