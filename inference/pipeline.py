from abc import ABC, abstractmethod
import torch
from typing import Any, Dict, Optional

class BaseInferencePipeline(ABC):
    """
    Abstract Interface for End-to-End Generation Pipeline.
    Orchestrates prompt compilation, memory lookup, video transformer sampling, VAE decoding, and assembly.
    """

    @abstractmethod
    def generate(self, prompt: str, num_frames: int = 16, seed: Optional[int] = None) -> torch.Tensor:
        """
        Executes pipeline generation for a text prompt.
        """
        pass
