from __future__ import annotations

from importlib import resources
from importlib.abc import Traversable
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


def _load_yaml_dict_from_traversable(
    resource: Traversable,
    *,
    display_name: str,
) -> dict[str, Any]:
    try:
        with resource.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config: {display_name}\n{exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected top-level YAML mapping in config file: {display_name}"
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


def _builtin_registration_preset_resources() -> dict[str, Traversable]:
    preset_root = resources.files("atlasspace.presets.registration")
    builtin_resources: dict[str, Traversable] = {}

    for resource in preset_root.iterdir():
        if resource.name.endswith(".yaml"):
            builtin_resources[resource.name] = resource
            builtin_resources[resource.name.removesuffix(".yaml")] = resource

    return builtin_resources


def list_presets() -> list[str]:
    return sorted(
        key
        for key in _builtin_registration_preset_resources()
        if not key.endswith(".yaml")
    )


def _load_registration_parameters_builtin(
    resource: Traversable,
    *,
    display_name: str,
) -> RegistrationParametersConfig:
    data = _load_yaml_dict_from_traversable(resource, display_name=display_name)
    try:
        return RegistrationParametersConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Config validation failed for {display_name} as RegistrationParametersConfig.\n{exc}"
        ) from exc


def load_preset(
    path: str | Path,
) -> RegistrationParametersConfig:
    path_value = Path(path)
    if path_value.exists():
        return _load_yaml_model(path_value, RegistrationParametersConfig)

    builtin_resources = _builtin_registration_preset_resources()
    builtin_key = str(path)
    builtin_resource = builtin_resources.get(builtin_key)
    if builtin_resource is not None:
        return _load_registration_parameters_builtin(
            builtin_resource,
            display_name=f"builtin registration preset '{builtin_key}'",
        )

    available_names = ", ".join(list_presets())
    raise FileNotFoundError(
        "Registration preset was not recognized as a built-in preset name and "
        "does not point to an existing file.\n"
        f"Got: {path}\n"
        f"Available built-in presets: {available_names}"
    )


def load_registration_batch_config(path: str | Path) -> RegistrationBatchConfig:
    return _load_yaml_model(path, RegistrationBatchConfig)


def load_registration_sweep_config(path: str | Path) -> RegistrationSweepConfig:
    return _load_yaml_model(path, RegistrationSweepConfig)
