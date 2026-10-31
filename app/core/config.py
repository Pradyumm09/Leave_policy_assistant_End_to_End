from pathlib import Path
from typing import Any

import yaml


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config() -> dict[str, Any]:
    """
    Load application configuration from config.yaml.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


CONFIG = load_config()
