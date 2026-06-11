from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from .config_models import (
    ImageConfig,
    RegistrationBatchConfig,
    RegistrationParametersConfig,
    RegistrationSweepConfig,
)


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config path is not a file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config: {config_path}\n{exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected top-level YAML mapping in config file: {config_path}"
        )
    return loaded


def _load_yaml_model(path: str | Path, model_cls: type[BaseModel]) -> BaseModel:
    config_path = Path(path)
    data = load_yaml_dict(config_path)
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Config validation failed for {config_path} as {model_cls.__name__}.\n{exc}"
        ) from exc


def load_registration_parameters_config(
    path: str | Path,
) -> RegistrationParametersConfig:
    return _load_yaml_model(path, RegistrationParametersConfig)


def load_registration_batch_config(path: str | Path) -> RegistrationBatchConfig:
    return _load_yaml_model(path, RegistrationBatchConfig)


def load_registration_sweep_config(path: str | Path) -> RegistrationSweepConfig:
    return _load_yaml_model(path, RegistrationSweepConfig)
