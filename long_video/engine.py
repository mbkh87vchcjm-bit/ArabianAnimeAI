from abc import ABC, abstractmethod
import torch
from typing import Any, Dict, List, Optional

class BaseLongVideoEngine(ABC):
    """
    Abstract Interface for Long Video Engine.
    Coordinates chunked video synthesis, temporal sliding windows, and long-range frame context consistency.
    """

    @abstractmethod
    def generate_long_sequence(
        self,
        prompts: List[str],
        total_frames: int,
        chunk_size: int = 16,
        overlap: int = 4,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generates extended video frame tensor across multi-chunk sequences.
        """
        pass
