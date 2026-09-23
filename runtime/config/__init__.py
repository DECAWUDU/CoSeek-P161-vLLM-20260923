import os
import yaml
import argparse
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_config(config: dict, args: argparse.Namespace):
    for key, value in args.__dict__.items():
        if key in config and value is not None:
            config[key] = value
    for key in (
        "api_base",
        "api_key",
        "api_version",
        "observer_api_base",
        "observer_api_key",
        "observer_api_version",
    ):
        value = config.get(key)
        if isinstance(value, str) and value:
            config[key] = os.getenv(value, value)
        elif value == "":
            config[key] = None
    return config


# General config
CONFIG_DIR = Path(__file__).resolve().parent

general_config = load_config(str(CONFIG_DIR / "general.yaml"))

prompts_config = load_config(str(CONFIG_DIR / "prompts.yaml"))
