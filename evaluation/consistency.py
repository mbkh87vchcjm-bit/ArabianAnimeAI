from abc import ABC, abstractmethod
import torch
from typing import Dict, Any

class BaseConsistencyChecker(ABC):
    """
    Abstract Interface for Consistency Checker component.
    Evaluates identity consistency (character features across frames) and scene stability.
    """

    @abstractmethod
    def evaluate_character_consistency(
        self,
        generated_frames: torch.Tensor,
        reference_features: torch.Tensor
    ) -> float:
        """Computes similarity score between generated frames and character reference identity."""
        pass

    @abstractmethod
    def evaluate_temporal_coherence(self, generated_frames: torch.Tensor) -> float:
        """Computes temporal flicker/coherence score across sequential video frames."""
        pass
