from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

import nibabel as nib
from pydantic import BaseModel, Field

from atlasspace.config.image_models import ImageConfig, OrientationAlignmentMode
from atlasspace.config.space_models import SpaceDefinition
from atlasspace.image.reorientation import compute_reorientation_transform
from atlasspace.runtime.registration import RegistrationJob, RegistrationResult


REGISTRATION_RESULT_FILENAME = "registration_result.json"
REGISTRATION_RESULT_SCHEMA_VERSION = 1
REGISTRATION_PARAMETERS_FILENAME = "registration_parameters.yaml"


class RegistrationManifestImage(BaseModel):
    image_id: str
    image: str
    space: SpaceDefinition
    normalized_image: str | None = None
    segmentations: dict[str, str] = Field(default_factory=dict)


class LegacyMigrationMetadata(BaseModel):
    source: Literal["legacy_registration_summary"] = "legacy_registration_summary"
    assumed_default_axis_labels: bool = True
    assumed_default_units: bool = True
    original_fixed_image: str | None = None
    original_moving_image: str | None = None


class RegistrationResultManifest(BaseModel):
    schema_version: Literal[1] = REGISTRATION_RESULT_SCHEMA_VERSION
    success: bool
    preset_name: str
    parameters_snapshot: str | None = REGISTRATION_PARAMETERS_FILENAME
    orientation_alignment: OrientationAlignmentMode
    fixed_image: RegistrationManifestImage
    moving_image: RegistrationManifestImage
    effective_fixed_space: SpaceDefinition
    effective_moving_space: SpaceDefinition
    runtime_seconds: float | None = None
    warped_image: str | None = None
    inverse_warped_image: str | None = None
    forward_transforms: list[str] = Field(default_factory=list)
    inverse_transforms: list[str] = Field(default_factory=list)
    transformed_segmentations: dict[str, str] = Field(default_factory=dict)
    error_message: str | None = None
    migration: LegacyMigrationMetadata | None = None


def _path_for_manifest(path: Path | None, output_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return str(path)


def _image_for_manifest(
    image_config: ImageConfig,
    normalized_image: Path | None,
    output_dir: Path,
) -> RegistrationManifestImage:
    return RegistrationManifestImage(
        image_id=image_config.image_id,
        image=str(image_config.image),
        space=image_config.space,
        normalized_image=_path_for_manifest(normalized_image, output_dir),
        segmentations={
            segmentation_id: str(path)
            for segmentation_id, path in image_config.segmentations.items()
        },
    )


def build_registration_result_manifest(
    job: RegistrationJob,
    result: RegistrationResult,
    *,
    fixed_normalized_path: Path | None,
    moving_normalized_path: Path | None,
    migration: LegacyMigrationMetadata | None = None,
) -> RegistrationResultManifest:
    output_dir = result.output_dir
    return RegistrationResultManifest(
        success=result.success,
        preset_name=result.preset_name,
        orientation_alignment=job.orientation_alignment,
        fixed_image=_image_for_manifest(
            job.fixed_image_config,
            fixed_normalized_path,
            output_dir,
        ),
        moving_image=_image_for_manifest(
            job.moving_image_config,
            moving_normalized_path,
            output_dir,
        ),
        effective_fixed_space=result.effective_fixed_space,
        effective_moving_space=result.effective_moving_space,
        runtime_seconds=result.runtime_seconds,
        warped_image=_path_for_manifest(result.warped_image, output_dir),
        inverse_warped_image=_path_for_manifest(result.inverse_warped_image, output_dir),
        forward_transforms=[
            _path_for_manifest(path, output_dir) for path in result.forward_transforms
        ],
        inverse_transforms=[
            _path_for_manifest(path, output_dir) for path in result.inverse_transforms
        ],
        transformed_segmentations={
            segmentation_id: _path_for_manifest(path, output_dir)
            for segmentation_id, path in result.transformed_segmentations.items()
        },
        error_message=result.error_message,
        migration=migration,
    )


def write_registration_result_manifest(
    job: RegistrationJob,
    result: RegistrationResult,
    *,
    fixed_normalized_path: Path | None,
    moving_normalized_path: Path | None,
    overwrite: bool = True,
) -> Path:
    manifest = build_registration_result_manifest(
        job,
        result,
        fixed_normalized_path=fixed_normalized_path,
        moving_normalized_path=moving_normalized_path,
    )
    return _write_manifest(manifest, result.output_dir, overwrite=overwrite)


def _write_manifest(
    manifest: RegistrationResultManifest,
    output_dir: Path,
    *,
    overwrite: bool,
) -> Path:
    manifest_path = output_dir / REGISTRATION_RESULT_FILENAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Registration result manifest already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_registration_result_manifest(
    output_dir: str | Path,
) -> RegistrationResultManifest:
    resolved_output_dir = Path(output_dir)
    manifest_path = resolved_output_dir / REGISTRATION_RESULT_FILENAME
    return RegistrationResultManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )


def _resolve_output_path(output_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else output_dir / path


def load_registration_result(output_dir: str | Path) -> RegistrationResult:
    resolved_output_dir = Path(output_dir)
    manifest = load_registration_result_manifest(resolved_output_dir)
    return RegistrationResult(
        fixed_image_id=manifest.fixed_image.image_id,
        moving_image_id=manifest.moving_image.image_id,
        preset_name=manifest.preset_name,
        output_dir=resolved_output_dir,
        success=manifest.success,
        declared_fixed_space=manifest.fixed_image.space,
        declared_moving_space=manifest.moving_image.space,
        effective_fixed_space=manifest.effective_fixed_space,
        effective_moving_space=manifest.effective_moving_space,
        runtime_seconds=manifest.runtime_seconds,
        warped_image=_resolve_output_path(resolved_output_dir, manifest.warped_image),
        inverse_warped_image=_resolve_output_path(
            resolved_output_dir,
            manifest.inverse_warped_image,
        ),
        forward_transforms=[
            _resolve_output_path(resolved_output_dir, path)
            for path in manifest.forward_transforms
        ],
        inverse_transforms=[
            _resolve_output_path(resolved_output_dir, path)
            for path in manifest.inverse_transforms
        ],
        transformed_segmentations={
            segmentation_id: _resolve_output_path(resolved_output_dir, path)
            for segmentation_id, path in manifest.transformed_segmentations.items()
        },
        error_message=manifest.error_message,
    )


def _parse_legacy_summary(summary_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(
        summary_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        if "=" not in line:
            raise ValueError(
                f"Malformed legacy summary line {line_number} in {summary_path}."
            )
        key, value = line.split("=", maxsplit=1)
        values[key] = value
    return values


def _required_legacy_value(values: dict[str, str], key: str) -> str:
    try:
        return values[key]
    except KeyError as exc:
        raise ValueError(f"Legacy registration summary is missing '{key}'.") from exc


def _parse_triplet(
    values: dict[str, str],
    key: str,
    value_type: type[float] | type[int],
) -> tuple:
    parsed = ast.literal_eval(_required_legacy_value(values, key))
    if not isinstance(parsed, tuple) or len(parsed) != 3:
        raise ValueError(f"Legacy registration summary '{key}' must be a triplet.")
    return tuple(value_type(value) for value in parsed)


def _nifti_shape(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Required normalized registration image not found: {path}")
    shape = tuple(int(value) for value in nib.load(str(path)).shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D normalized registration image at {path}.")
    return shape


def _declared_shape_from_effective(
    effective_shape: tuple[int, int, int],
    declared_orientation: str,
    effective_orientation: str,
) -> tuple[int, int, int]:
    permutation, _ = compute_reorientation_transform(
        declared_orientation,
        effective_orientation,
    )
    declared_shape = [0, 0, 0]
    for effective_axis, declared_axis in enumerate(permutation):
        declared_shape[declared_axis] = effective_shape[effective_axis]
    return tuple(declared_shape)


def _legacy_spaces(
    values: dict[str, str],
    role: Literal["fixed", "moving"],
    effective_shape: tuple[int, int, int],
) -> tuple[SpaceDefinition, SpaceDefinition]:
    declared_orientation = _required_legacy_value(
        values,
        f"configured_{role}_orientation",
    )
    effective_orientation = _required_legacy_value(
        values,
        f"effective_{role}_orientation",
    )
    declared_resolution = _parse_triplet(
        values,
        f"configured_{role}_resolution_um",
        float,
    )
    effective_resolution = _parse_triplet(
        values,
        f"effective_{role}_resolution_um",
        float,
    )
    declared_shape = _declared_shape_from_effective(
        effective_shape,
        declared_orientation,
        effective_orientation,
    )
    declared_space = SpaceDefinition(
        space_name=_required_legacy_value(values, f"{role}_space_name"),
        orientation=declared_orientation,
        resolution_um=declared_resolution,
        shape=declared_shape,
    )
    permutation, _ = compute_reorientation_transform(
        declared_orientation,
        effective_orientation,
    )
    effective_space = SpaceDefinition(
        space_name=declared_space.space_name,
        orientation=effective_orientation,
        axis_labels=tuple(
            declared_space.axis_labels[declared_axis]
            for declared_axis in permutation
        ),
        units=declared_space.units,
        resolution_um=effective_resolution,
        shape=effective_shape,
    )
    return declared_space, effective_space


def _infer_orientation_alignment(
    declared_fixed: SpaceDefinition,
    effective_fixed: SpaceDefinition,
    declared_moving: SpaceDefinition,
    effective_moving: SpaceDefinition,
) -> OrientationAlignmentMode:
    fixed_changed = declared_fixed.orientation != effective_fixed.orientation
    moving_changed = declared_moving.orientation != effective_moving.orientation
    if fixed_changed and not moving_changed:
        return "fixed_to_moving"
    if moving_changed and not fixed_changed:
        return "moving_to_fixed"
    if not fixed_changed and not moving_changed:
        return "none"
    raise ValueError(
        "Could not infer orientation alignment because both legacy inputs changed orientation."
    )


def _existing_relative_path(output_dir: Path, filename: str) -> str | None:
    path = output_dir / filename
    return filename if path.exists() else None


def _legacy_transform_paths(
    output_dir: Path,
    *,
    success: bool,
) -> tuple[list[str], list[str]]:
    affine = output_dir / "ANTsPy_0GenericAffine.mat"
    if success and not affine.exists():
        raise FileNotFoundError(f"Required ANTs affine transform not found: {affine}")
    if not affine.exists():
        return [], []

    forward = [affine.name]
    warp = output_dir / "ANTsPy_1Warp.nii.gz"
    if warp.exists():
        forward.insert(0, warp.name)

    inverse = [affine.name]
    inverse_warp = output_dir / "ANTsPy_1InverseWarp.nii.gz"
    if inverse_warp.exists():
        inverse.append(inverse_warp.name)
    return forward, inverse


def build_legacy_registration_result_manifest(
    output_dir: str | Path,
    *,
    fixed_image: str | Path | None = None,
    moving_image: str | Path | None = None,
) -> RegistrationResultManifest:
    resolved_output_dir = Path(output_dir)
    summary_path = resolved_output_dir / "registration_summary.txt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Legacy registration summary not found: {summary_path}")

    values = _parse_legacy_summary(summary_path)
    success_value = _required_legacy_value(values, "success")
    if success_value not in {"True", "False"}:
        raise ValueError("Legacy registration summary 'success' must be True or False.")
    success = success_value == "True"

    fixed_normalized = resolved_output_dir / "fixed_normalized_for_registration.nii.gz"
    moving_normalized = resolved_output_dir / "moving_normalized_for_registration.nii.gz"
    declared_fixed, effective_fixed = _legacy_spaces(
        values,
        "fixed",
        _nifti_shape(fixed_normalized),
    )
    declared_moving, effective_moving = _legacy_spaces(
        values,
        "moving",
        _nifti_shape(moving_normalized),
    )
    orientation_alignment = _infer_orientation_alignment(
        declared_fixed,
        effective_fixed,
        declared_moving,
        effective_moving,
    )
    forward_transforms, inverse_transforms = _legacy_transform_paths(
        resolved_output_dir,
        success=success,
    )
    transformed_segmentations: dict[str, str] = {}
    for path in resolved_output_dir.rglob("*_WarpedSegmentation.nii.gz"):
        segmentation_id = path.name.removesuffix("_WarpedSegmentation.nii.gz")
        if segmentation_id in transformed_segmentations:
            raise ValueError(
                "Multiple legacy transformed segmentations use the id "
                f"'{segmentation_id}' in {resolved_output_dir}."
            )
        transformed_segmentations[segmentation_id] = path.relative_to(
            resolved_output_dir
        ).as_posix()
    error_value = values.get("error_message")
    error_message = None if error_value in {None, "None"} else error_value
    runtime_value = values.get("runtime_seconds")
    legacy_fixed_image = _required_legacy_value(values, "fixed_image")
    legacy_moving_image = _required_legacy_value(values, "moving_image")
    resolved_fixed_image = str(fixed_image) if fixed_image is not None else legacy_fixed_image
    resolved_moving_image = (
        str(moving_image) if moving_image is not None else legacy_moving_image
    )

    return RegistrationResultManifest(
        success=success,
        preset_name=_required_legacy_value(values, "preset_name"),
        parameters_snapshot=_existing_relative_path(
            resolved_output_dir,
            REGISTRATION_PARAMETERS_FILENAME,
        ),
        orientation_alignment=orientation_alignment,
        fixed_image=RegistrationManifestImage(
            image_id=_required_legacy_value(values, "fixed_image_id"),
            image=resolved_fixed_image,
            space=declared_fixed,
            normalized_image=fixed_normalized.name,
        ),
        moving_image=RegistrationManifestImage(
            image_id=_required_legacy_value(values, "moving_image_id"),
            image=resolved_moving_image,
            space=declared_moving,
            normalized_image=moving_normalized.name,
        ),
        effective_fixed_space=effective_fixed,
        effective_moving_space=effective_moving,
        runtime_seconds=float(runtime_value) if runtime_value is not None else None,
        warped_image=_existing_relative_path(resolved_output_dir, "ANTsPy_Warped.nii.gz"),
        inverse_warped_image=_existing_relative_path(
            resolved_output_dir,
            "ANTsPy_InverseWarped.nii.gz",
        ),
        forward_transforms=forward_transforms,
        inverse_transforms=inverse_transforms,
        transformed_segmentations=transformed_segmentations,
        error_message=error_message,
        migration=LegacyMigrationMetadata(
            original_fixed_image=(
                legacy_fixed_image if resolved_fixed_image != legacy_fixed_image else None
            ),
            original_moving_image=(
                legacy_moving_image if resolved_moving_image != legacy_moving_image else None
            ),
        ),
    )


def migrate_legacy_registration_output(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    fixed_image: str | Path | None = None,
    moving_image: str | Path | None = None,
) -> Path:
    resolved_output_dir = Path(output_dir)
    manifest_path = resolved_output_dir / REGISTRATION_RESULT_FILENAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Registration result manifest already exists: {manifest_path}")
    manifest = build_legacy_registration_result_manifest(
        resolved_output_dir,
        fixed_image=fixed_image,
        moving_image=moving_image,
    )
    return _write_manifest(manifest, resolved_output_dir, overwrite=overwrite)
