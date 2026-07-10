from pathlib import Path
import yaml
from types import SimpleNamespace


def dict_to_namespace(d):
    """
    Convert nested dictionaries into dot-accessible objects.
    Example:
        cfg.learning.lr instead of cfg["learning"]["lr"]
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def load_config(config_path="configs/default.yaml"):
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return dict_to_namespace(cfg_dict)