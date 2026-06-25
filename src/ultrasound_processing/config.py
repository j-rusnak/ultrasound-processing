from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing required structure."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate(config: dict[str, Any]) -> None:
    required_sections = ("data", "model", "training", "artifacts")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(f"missing required section(s): {', '.join(missing)}")


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(
    config_path: str | Path | None = None,
    *,
    default_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a YAML config and optionally merge it over a default config."""

    if config_path is None and default_path is None:
        config_path = default_config_path()

    config: dict[str, Any] = {}
    if default_path is not None:
        config = _read_yaml(Path(default_path))

    if config_path is not None:
        config = _deep_merge(config, _read_yaml(Path(config_path)))

    _validate(config)
    return config


def resolve_project_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    return root / value
