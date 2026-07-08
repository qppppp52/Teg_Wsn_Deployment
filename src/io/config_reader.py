"""Read and merge YAML configuration files."""
from __future__ import annotations

import copy
import os
import yaml


DEFAULT_CONFIG_NAMES = [
    "default", "scene", "teg", "channel", "node", "heatsink",
    "constraints", "mode", "dqn", "drl_init", "experiment",
]


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_dir: str = "configs") -> dict:
    """Load the default config stack from a config directory."""
    cfg = {}
    for name in DEFAULT_CONFIG_NAMES:
        path = os.path.join(config_dir, f"{name}.yaml")
        if os.path.exists(path):
            cfg = _deep_update(cfg, _load_yaml(path))
    return cfg


def load_experiment_config(experiment_path: str) -> dict:
    """Load base configs and then apply an experiment config_file mapping."""
    experiment_path = os.path.abspath(experiment_path)
    project_root = os.path.dirname(os.path.dirname(experiment_path)) if os.path.basename(os.path.dirname(experiment_path)) == "configs" else os.getcwd()
    config_dir = os.path.join(project_root, "configs")
    cfg = load_config(config_dir)
    exp_cfg = _load_yaml(experiment_path)

    for _, file_path in exp_cfg.get("config_files", {}).items():
        resolved = file_path
        if not os.path.isabs(resolved):
            resolved = os.path.join(project_root, file_path)
        if os.path.exists(resolved):
            cfg = _deep_update(cfg, _load_yaml(resolved))
    cfg = _deep_update(cfg, exp_cfg)
    cfg["project_root"] = project_root
    return cfg


def flatten_config(cfg: dict) -> dict:
    """Flatten one-level nested config keys for tabular logging."""
    flat = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k}.{sk}"] = sv
        else:
            flat[k] = v
    return flat
