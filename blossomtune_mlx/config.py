"""blossomtunellm-mlx: A Flower client app for federated learning with MLX."""

from omegaconf import DictConfig

from flwr.common import Context
from flwr.common.config import unflatten_dict


def replace_keys(input_dict, match: str = "-", target: str = "_") -> dict:
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict


def get_run_config(context: Context) -> DictConfig:
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    return cfg
