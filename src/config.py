"""Load config.yaml and resolve paths relative to the project root."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "config.yaml") -> dict:
    """Read the YAML config and turn every entry under `paths` into an absolute Path."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {key: (PROJECT_ROOT / value).resolve() for key, value in cfg["paths"].items()}
    return cfg
