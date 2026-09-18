import os
import yaml
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        self.config_path = config_path
        self.config_data = self._load_yaml(config_path)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        val = self.config_data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def load_pipeline_config(self, experiment_path: str) -> Dict[str, Any]:
        exp_data = self._load_yaml(experiment_path)
        merged = self.config_data.copy()
        merged.update(exp_data)
        return merged
