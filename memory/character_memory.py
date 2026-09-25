from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch

class BaseCharacterMemory(ABC):
    """
    Abstract Interface for Character Memory component.
    Stores key character reference embeddings, visual features, and identity vectors across scenes and shots.
    """

    @abstractmethod
    def store_character(self, character_id: str, feature_vector: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """Stores or updates a character profile in memory."""
        pass

    @abstractmethod
    def get_character(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves character profile and feature vectors."""
        pass

    @abstractmethod
    def list_characters(self) -> list[str]:
        """Lists all stored character IDs."""
        pass
