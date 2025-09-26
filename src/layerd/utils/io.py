import json
from typing import Any

import yaml


def read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_json(path: str) -> dict[str, Any] | list[Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict[str, Any] | list[Any], path: str, indent: int | None = None) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def save_yaml(data: dict[str, Any] | list[Any], path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)
