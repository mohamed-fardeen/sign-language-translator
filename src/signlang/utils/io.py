from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_yaml(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(obj: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p