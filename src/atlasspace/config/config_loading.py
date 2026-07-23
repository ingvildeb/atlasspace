from __future__ import annotations

from importlib import resources
from importlib.abc import Traversable
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import yaml
from pydantic import BaseModel, ValidationError

from atlasspace.config.image_models import ImageConfig
from atlasspace.config.job_spec_models import (
    RegistrationJobSpecConfig,
    RegistrationPair,
    RegistrationPlan,
)
from atlasspace.config.preset_models import RegistrationParametersConfig
from atlasspace.config.space_models import SpaceDefinition


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


def load_toml_dict(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config path is not a file: {config_path}")

    try:
        with config_path.open("rb") as handle:
            loaded = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Failed to parse TOML config: {config_path}\n{exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected top-level TOML mapping in config file: {config_path}"
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


def _validate_model(
    data: dict[str, Any],
    model_cls: type[BaseModel],
    *,
    source_label: str,
) -> BaseModel:
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Config validation failed for {source_label} as {model_cls.__name__}.\n{exc}"
        ) from exc


def _load_yaml_model(path: str | Path, model_cls: type[BaseModel]) -> BaseModel:
    config_path = Path(path)
    data = load_yaml_dict(config_path)
    return _validate_model(data, model_cls, source_label=str(config_path))


def _load_toml_model(path: str | Path, model_cls: type[BaseModel]) -> BaseModel:
    config_path = Path(path)
    data = load_toml_dict(config_path)
    return _validate_model(data, model_cls, source_label=str(config_path))


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


def load_preset(path: str | Path) -> RegistrationParametersConfig:
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


def load_registration_job_spec_config(path: str | Path) -> RegistrationJobSpecConfig:
    return _load_toml_model(path, RegistrationJobSpecConfig)


def load_registration_plan_from_dict(
    data: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> RegistrationPlan:
    job_spec = _validate_model(
        data,
        RegistrationJobSpecConfig,
        source_label=str(config_path) if config_path is not None else "<dict>",
    )
    return normalize_registration_job_spec(
        job_spec,
        config_path=config_path,
    )


def load_registration_plan(path: str | Path) -> RegistrationPlan:
    config_path = Path(path)
    return load_registration_plan_from_dict(
        load_toml_dict(config_path),
        config_path=config_path,
    )


def normalize_registration_job_spec(
    job_spec: RegistrationJobSpecConfig,
    *,
    config_path: str | Path | None = None,
) -> RegistrationPlan:
    base_dir = Path(config_path).parent if config_path is not None else None
    resolved_images = _resolve_job_spec_images(job_spec, base_dir=base_dir)
    mode = job_spec.mode_name()
    preset_references = [
        _resolve_preset_reference(preset_reference, base_dir=base_dir)
        for preset_reference in job_spec.run.registration_presets
    ]

    if mode == "single":
        assert job_spec.single is not None
        single_output_dir = _resolve_path(job_spec.run.output_dir, base_dir=base_dir)
        return RegistrationPlan(
            mode="single",
            preset_references=preset_references,
            orientation_alignment=job_spec.run.orientation_alignment,
            write_input_images=job_spec.run.write_input_images,
            single_output_dir=single_output_dir,
            images=resolved_images,
            pairs=[
                _build_registration_pair(
                    fixed_image_id=job_spec.single.fixed_image,
                    moving_image_id=job_spec.single.moving_image,
                )
            ],
            moving_segmentations=job_spec.moving_segmentations,
        )

    if mode == "batch":
        assert job_spec.batch is not None
        pairs = [
            _resolve_batch_pair(
                run_image_id=run_image_id,
                template_image_id=template_image_id,
                template_role=job_spec.batch.template_role,
            )
            for run_image_id, template_image_id in job_spec.batch.image_to_template.items()
        ]
        return RegistrationPlan(
            mode="batch",
            preset_references=preset_references,
            orientation_alignment=job_spec.run.orientation_alignment,
            write_input_images=job_spec.run.write_input_images,
            batch_output_subdir=job_spec.run.output_subdir,
            output_root=_resolve_path(job_spec.run.output_root, base_dir=base_dir),
            images=resolved_images,
            pairs=pairs,
            moving_segmentations=job_spec.moving_segmentations,
        )

    assert job_spec.sweep is not None
    pairs = [
        _resolve_sweep_pair(
            shared_image_id=job_spec.sweep.shared_image,
            run_image_id=run_image_id,
            shared_image_role=job_spec.sweep.shared_image_role,
        )
        for run_image_id in job_spec.sweep.run_images
    ]
    return RegistrationPlan(
        mode="sweep",
        preset_references=preset_references,
        orientation_alignment=job_spec.run.orientation_alignment,
        write_input_images=job_spec.run.write_input_images,
        output_root=_resolve_path(job_spec.run.output_root, base_dir=base_dir),
        images=resolved_images,
        pairs=pairs,
        moving_segmentations=job_spec.moving_segmentations,
    )


def _resolve_job_spec_images(
    job_spec: RegistrationJobSpecConfig,
    *,
    base_dir: Path | None,
) -> dict[str, ImageConfig]:
    resolved_images: dict[str, ImageConfig] = {}
    default_orientation = job_spec.image_defaults.orientation
    default_resolution_um = job_spec.image_defaults.resolution_um

    for image_id, image_config in job_spec.images.items():
        orientation = image_config.orientation or default_orientation
        if orientation is None:
            raise ValueError(
                f"Image '{image_id}' is missing orientation and no [image_defaults].orientation was provided."
            )

        resolution_um = image_config.resolution_um
        if resolution_um is None:
            resolution_um = default_resolution_um
        if resolution_um is None:
            raise ValueError(
                f"Image '{image_id}' is missing resolution_um and no [image_defaults].resolution_um was provided."
            )

        resolved_images[image_id] = ImageConfig(
            image_id=image_id,
            image=_resolve_path(image_config.image, base_dir=base_dir),
            space=SpaceDefinition(
                space_name=image_config.space_name or image_id,
                orientation=orientation,
                resolution_um=_isotropic_resolution_um(resolution_um),
            ),
            segmentations={
                segmentation_id: _resolve_path(segmentation_path, base_dir=base_dir)
                for segmentation_id, segmentation_path in image_config.segmentations.items()
            },
        )

    return resolved_images


def _resolve_batch_pair(
    *,
    run_image_id: str,
    template_image_id: str,
    template_role: str,
) -> RegistrationPair:
    if template_role == "moving":
        return _build_registration_pair(
            fixed_image_id=run_image_id,
            moving_image_id=template_image_id,
            run_image_id=run_image_id,
        )
    if template_role == "fixed":
        return _build_registration_pair(
            fixed_image_id=template_image_id,
            moving_image_id=run_image_id,
            run_image_id=run_image_id,
        )
    raise ValueError(f"Unsupported template_role: {template_role}")


def _resolve_sweep_pair(
    *,
    shared_image_id: str,
    run_image_id: str,
    shared_image_role: str,
) -> RegistrationPair:
    if shared_image_role == "fixed":
        return _build_registration_pair(
            fixed_image_id=shared_image_id,
            moving_image_id=run_image_id,
            run_image_id=run_image_id,
        )
    if shared_image_role == "moving":
        return _build_registration_pair(
            fixed_image_id=run_image_id,
            moving_image_id=shared_image_id,
            run_image_id=run_image_id,
        )
    raise ValueError(f"Unsupported shared_image_role: {shared_image_role}")


def _build_registration_pair(
    *,
    fixed_image_id: str,
    moving_image_id: str,
    run_image_id: str | None = None,
) -> RegistrationPair:
    return RegistrationPair(
        fixed_image_id=fixed_image_id,
        moving_image_id=moving_image_id,
        run_image_id=run_image_id,
    )


def _resolve_preset_reference(
    preset_reference: str,
    *,
    base_dir: Path | None,
) -> str:
    candidate = Path(preset_reference)
    if base_dir is None:
        return preset_reference
    if not _looks_like_path_reference(preset_reference):
        return preset_reference
    return str(_resolve_path(candidate, base_dir=base_dir))


def _looks_like_path_reference(value: str) -> bool:
    candidate = Path(value)
    return candidate.is_absolute() or bool(candidate.suffix) or any(
        separator in value for separator in ("/", "\\")
    )


def _resolve_path(
    path_value: Path | None,
    *,
    base_dir: Path | None,
) -> Path | None:
    if path_value is None or base_dir is None or path_value.is_absolute():
        return path_value
    return base_dir / path_value


def _isotropic_resolution_um(value: float) -> tuple[float, float, float]:
    return (value, value, value)
