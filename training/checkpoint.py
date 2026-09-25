from dataclasses import dataclass, field, asdict
import json
import os
import time
from typing import Any, Dict, List, Optional
import torch

@dataclass
class CheckpointMetadata:
    """
    Structured metadata for model checkpoints, tracking training step, epoch, loss metrics, and timestamp.
    """
    step: int
    epoch: int
    loss: float
    experiment_name: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    metrics: Dict[str, float] = field(default_factory=dict)
    model_version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        return cls(**data)


class CheckpointManager:
    """
    Abstracts saving and loading PyTorch checkpoints alongside structured metadata.
    Prevents loss of successful experiments and checkpoints.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        filename: str,
        state_dict: Dict[str, Any],
        metadata: CheckpointMetadata,
        optimizer_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Saves checkpoint tensor state dict, optimizer state, and metadata.
        """
        if not filename.endswith(".pt") and not filename.endswith(".pth"):
            filename = f"{filename}.pt"

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        meta_path = checkpoint_path + ".json"

        checkpoint_payload = {
            "state_dict": state_dict,
            "optimizer_state": optimizer_state,
            "metadata": metadata.to_dict()
        }

        torch.save(checkpoint_payload, checkpoint_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return checkpoint_path

    def load_checkpoint(self, filename: str, map_location: str = "cpu") -> Dict[str, Any]:
        """
        Loads checkpoint state dict and metadata.
        """
        if not filename.endswith(".pt") and not filename.endswith(".pth"):
            filename = f"{filename}.pt"

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        return payload

    def load_metadata(self, filename: str) -> CheckpointMetadata:
        """
        Loads standalone metadata JSON file for a checkpoint.
        """
        if not filename.endswith(".pt") and not filename.endswith(".pth"):
            filename = f"{filename}.pt"

        meta_path = os.path.join(self.checkpoint_dir, filename + ".json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Checkpoint metadata file not found at: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return CheckpointMetadata.from_dict(data)

    def list_checkpoints(self) -> List[str]:
        """
        Lists all checkpoint filenames in the directory.
        """
        return [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt") or f.endswith(".pth")]
