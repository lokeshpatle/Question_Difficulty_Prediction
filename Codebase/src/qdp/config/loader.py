from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from qdp.config.schema import validate_config


def load_yaml(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_update(base: dict, override: dict):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(project_root: Path, path: str = "config/default.yaml", override: dict | None = None):
    config = load_yaml(Path(project_root) / path)
    if override:
        config = deep_update(config, override)
    validate_config(config)
    return config
