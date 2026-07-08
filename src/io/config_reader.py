"""读取和合并YAML配置文件"""
import os
import yaml


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str = "configs") -> dict:
    cfg = {}
    files = ["default", "scene", "teg", "channel", "node",
             "heatsink", "constraints", "mode", "experiment"]
    for name in files:
        path = os.path.join(config_dir, f"{name}.yaml")
        if os.path.exists(path):
            cfg.update(_load_yaml(path))
    return cfg


def flatten_config(cfg: dict) -> dict:
    """展开嵌套 dict 为扁平键名"""
    flat = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k}.{sk}"] = sv
        else:
            flat[k] = v
    return flat
