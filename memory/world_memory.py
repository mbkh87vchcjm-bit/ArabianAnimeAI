from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch

class BaseWorldMemory(ABC):
    """
    Abstract Interface for World / Location Memory component.
    Stores key environment, background, architectural, and scene lighting representations.
    """

    @abstractmethod
    def store_location(self, location_id: str, feature_vector: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """Stores or updates a location profile in memory."""
        pass

    @abstractmethod
    def get_location(self, location_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves location profile and feature vectors."""
        pass

    @abstractmethod
    def list_locations(self) -> list[str]:
        """Lists all stored location IDs."""
        pass
