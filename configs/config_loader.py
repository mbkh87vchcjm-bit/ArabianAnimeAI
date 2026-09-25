import os
from typing import Any, Dict, Optional, Union
import yaml

class Config(dict):
    """
    Attribute-accessible dictionary wrapper for configuration dictionary.
    Allows dot-notation access (e.g., config.system.seed).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Configuration object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config and nested Config objects back to a standard dictionary."""
        result = {}
        for key, value in self.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str = "configs/default.yaml", overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load a YAML configuration file and apply optional dictionary overrides.

    Args:
        config_path (str): Path to the YAML configuration file.
        overrides (Dict[str, Any], optional): Dictionary of overrides to merge.

    Returns:
        Config: Config object with attribute access.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    config = Config(data)

    if overrides:
        _merge_dict(config, overrides)

    return config


def _merge_dict(target: Config, source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], Config):
            _merge_dict(target[key], value)
        else:
            target[key] = value
