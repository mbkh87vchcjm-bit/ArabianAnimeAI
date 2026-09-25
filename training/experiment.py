import json
import os
import time
import uuid
from typing import Any, Dict, Optional
import yaml
from configs.config_loader import Config

class ExperimentTracker:
    """
    Manages experiment run directories, configuration snapshots, and run metadata.
    Ensures every experiment run is isolated and preserved without losing previous runs.
    """

    def __init__(self, experiment_name: str, base_dir: str = "experiments", config: Optional[Config] = None):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Add microsecond timestamp component and short UUID hex fragment to prevent run ID collisions
        microseconds = f"{int((time.time() % 1) * 1_000_000):06d}"
        short_uuid = uuid.uuid4().hex[:6]
        self.run_id = f"{experiment_name}_{self.timestamp}_{microseconds}_{short_uuid}"
        self.experiment_dir = os.path.join(base_dir, self.run_id)
        self.checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.logs_dir = os.path.join(self.experiment_dir, "logs")

        self._setup_directories()

        if config is not None:
            self.save_config_snapshot(config)

    def _setup_directories(self) -> None:
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def save_config_snapshot(self, config: Config) -> str:
        snapshot_path = os.path.join(self.experiment_dir, "config_snapshot.yaml")
        config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
        return snapshot_path

    def save_metadata(self, metadata: Dict[str, Any]) -> str:
        metadata_path = os.path.join(self.experiment_dir, "metadata.json")
        data = {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            **metadata
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return metadata_path

    def get_checkpoint_dir(self) -> str:
        return self.checkpoints_dir

    def get_experiment_dir(self) -> str:
        return self.experiment_dir
