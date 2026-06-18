from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from atlasspace.config.config_models import ImageConfig
from atlasspace.config.space_models import SpaceDefinition
from atlasspace.image.reorientation import reorient_array_to_match
from atlasspace.io.nifti import write_nifti_from_array
from atlasspace.runtime.transforms import TransformDirection, TransformSequence
from atlasspace.transforms.antspy_transformation import (
    ants,
    apply_transform_paths_to_image,
    apply_transform_paths_to_points,
)


def transform_image(
    image_config: ImageConfig,
    transform_sequence: TransformSequence,
    reference_config: ImageConfig,
    *,
    direction: TransformDirection = "forward",
    interpolation: str = "linear",
    output_path: Path | None = None,
    output_dir: Path | None = None,
    write_intermediates: bool = False,
) -> ImageConfig:
    resolved_output_path = resolve_transform_output_path(
        image_config,
        transform_sequence,
        direction=direction,
        output_path=output_path,
        output_dir=output_dir,
    )
    prepared_input_path = (
        _derived_output_path(resolved_output_path, "prepared_input")
        if write_intermediates
        else resolved_output_path.parent / f"{resolved_output_path.name}.prepared_input.tmp.nii.gz"
    )
    prepared_reference_path = (
        _derived_output_path(resolved_output_path, "prepared_reference")
        if write_intermediates
        else resolved_output_path.parent / f"{resolved_output_path.name}.prepared_reference.tmp.nii.gz"
    )

    prepared_image_config = prepare_image_for_transform(
        image_config,
        transform_sequence,
        direction=direction,
        output_path=prepared_input_path,
    )
    prepared_reference_config = prepare_reference_for_transform(
        reference_config,
        transform_sequence,
        direction=direction,
        output_path=prepared_reference_path,
    )
    transformed_in_transform_space = _transform_image_in_transform_space(
        prepared_image_config,
        transform_sequence,
        prepared_reference_config,
        direction=direction,
        interpolation=interpolation,
        output_path=resolved_output_path,
        dtype=np.float32,
        image_id_suffix=f"{direction}_transformed_in_transform_space",
    )
    try:
        return transformed_in_transform_space
    finally:
        if not write_intermediates:
            _cleanup_intermediate_paths(
                prepared_input_path,
                prepared_reference_path,
            )


def transform_segmentation(
    image_config: ImageConfig,
    transform_sequence: TransformSequence,
    reference_config: ImageConfig,
    *,
    direction: TransformDirection = "forward",
    interpolation: str = "nearestNeighbor",
    output_path: Path | None = None,
    output_dir: Path | None = None,
    write_intermediates: bool = False,
) -> ImageConfig:
    resolved_output_path = resolve_transform_output_path(
        image_config,
        transform_sequence,
        direction=direction,
        output_path=output_path,
        output_dir=output_dir,
        kind="segmentation",
    )
    prepared_input_path = (
        _derived_output_path(resolved_output_path, "prepared_input")
        if write_intermediates
        else resolved_output_path.parent / f"{resolved_output_path.name}.prepared_input.tmp.nii.gz"
    )
    prepared_reference_path = (
        _derived_output_path(resolved_output_path, "prepared_reference")
        if write_intermediates
        else resolved_output_path.parent / f"{resolved_output_path.name}.prepared_reference.tmp.nii.gz"
    )

    prepared_image_config = prepare_image_for_transform(
        image_config,
        transform_sequence,
        direction=direction,
        output_path=prepared_input_path,
    )
    prepared_reference_config = prepare_reference_for_transform(
        reference_config,
        transform_sequence,
        direction=direction,
        output_path=prepared_reference_path,
    )
    transformed_in_transform_space = _transform_image_in_transform_space(
        prepared_image_config,
        transform_sequence,
        prepared_reference_config,
        direction=direction,
        interpolation=interpolation,
        output_path=resolved_output_path,
        dtype=np.int32,
        image_id_suffix=f"{direction}_transformed_segmentation_in_transform_space",
    )
    try:
        return transformed_in_transform_space
    finally:
        if not write_intermediates:
            _cleanup_intermediate_paths(
                prepared_input_path,
                prepared_reference_path,
            )


def transform_points(
    points_xyz: np.ndarray,
    transform_sequence: TransformSequence,
    *,
    direction: TransformDirection = "forward",
) -> np.ndarray:
    transform_paths = transform_sequence.paths_for_direction(direction)
    return apply_transform_paths_to_points(points_xyz, transform_paths)


def resolve_transform_output_path(
    image_config: ImageConfig,
    transform_sequence: TransformSequence,
    *,
    direction: TransformDirection,
    output_path: Path | None = None,
    output_dir: Path | None = None,
    kind: str = "image",
) -> Path:
    if output_path is not None and output_dir is not None:
        raise ValueError("Provide either output_path or output_dir, not both.")
    if output_path is not None:
        return output_path

    destination_dir = output_dir or Path.cwd()
    return build_transform_output_path(
        image_config.space,
        destination_dir,
        direction=direction,
        kind=kind,
    )


def build_transform_output_path(
    output_space: SpaceDefinition,
    output_dir: Path,
    *,
    direction: TransformDirection,
    kind: str = "image",
) -> Path:
    space_name = (output_space.space_name or "space").strip() or "space"

    if direction == "forward":
        suffix = "Warped"
    elif direction == "inverse":
        suffix = "InverseWarped"
    else:
        raise ValueError(f"Unsupported transform direction: {direction}")

    if kind == "segmentation":
        suffix = f"{suffix}Segmentation"
    elif kind != "image":
        raise ValueError(f"Unsupported transform output kind: {kind}")

    return output_dir / f"{space_name}_{suffix}.nii.gz"


def prepare_image_for_transform(
    image_config: ImageConfig,
    transform_sequence: TransformSequence,
    *,
    direction: TransformDirection = "forward",
    output_path: Path,
) -> ImageConfig:
    declared_input_space, _ = _declared_spaces(transform_sequence, direction)
    transform_input_space, _ = _transform_spaces(transform_sequence, direction)
    if _spaces_are_compatible(image_config.space, declared_input_space):
        pass
    elif _spaces_are_compatible(image_config.space, transform_input_space):
        return _reorient_image_to_space(
            image_config,
            transform_input_space,
            output_path=output_path,
            image_id_suffix="prepared_for_transform",
        )
    else:
        raise ValueError(
            "image_config.space is incompatible with the transform input space. "
            "Expected either the declared input space or the effective transform input space."
        )
    return _reorient_image_to_space(
        image_config,
        transform_input_space,
        output_path=output_path,
        image_id_suffix="prepared_for_transform",
    )


def prepare_reference_for_transform(
    reference_config: ImageConfig,
    transform_sequence: TransformSequence,
    *,
    direction: TransformDirection = "forward",
    output_path: Path,
) -> ImageConfig:
    _, declared_output_space = _declared_spaces(transform_sequence, direction)
    _, transform_output_space = _transform_spaces(transform_sequence, direction)
    if _spaces_are_compatible(reference_config.space, declared_output_space):
        pass
    elif _spaces_are_compatible(reference_config.space, transform_output_space):
        return _reorient_image_to_space(
            reference_config,
            transform_output_space,
            output_path=output_path,
            image_id_suffix="prepared_reference_for_transform",
        )
    else:
        raise ValueError(
            "reference_config.space is incompatible with the transform output space. "
            "Expected either the declared output space or the effective transform output space."
        )
    return _reorient_image_to_space(
        reference_config,
        transform_output_space,
        output_path=output_path,
        image_id_suffix="prepared_reference_for_transform",
    )


def _transform_image_in_transform_space(
    image_config: ImageConfig,
    transform_sequence: TransformSequence,
    reference_config: ImageConfig,
    *,
    direction: TransformDirection,
    interpolation: str,
    output_path: Path,
    dtype,
    image_id_suffix: str,
) -> ImageConfig:
    expected_input_space, expected_output_space = _transform_spaces(
        transform_sequence,
        direction,
    )
    _validate_space_compatibility(
        actual=image_config.space,
        expected=expected_input_space,
        label="prepared image space",
    )
    _validate_space_compatibility(
        actual=reference_config.space,
        expected=expected_output_space,
        label="prepared reference space",
    )

    moving_image = ants.image_read(str(image_config.image))
    reference_image = ants.image_read(str(reference_config.image))
    transformed_image = apply_transform_paths_to_image(
        moving_image,
        reference_image,
        transform_sequence.paths_for_direction(direction),
        interpolation=interpolation,
    )

    transformed_array = transformed_image.numpy()
    write_nifti_from_array(
        transformed_array,
        reference_config.space,
        output_path,
        dtype=dtype,
    )

    return ImageConfig(
        image_id=f"{image_config.image_id}_{image_id_suffix}",
        image=output_path,
        space=reference_config.space,
    )


def _declared_spaces(
    transform_sequence: TransformSequence,
    direction: TransformDirection,
) -> tuple[SpaceDefinition, SpaceDefinition]:
    if direction == "forward":
        return transform_sequence.source_space, transform_sequence.target_space
    if direction == "inverse":
        return transform_sequence.target_space, transform_sequence.source_space
    raise ValueError(f"Unsupported transform direction: {direction}")


def _transform_spaces(
    transform_sequence: TransformSequence,
    direction: TransformDirection,
) -> tuple[SpaceDefinition, SpaceDefinition]:
    if direction == "forward":
        return (
            transform_sequence.source_transform_space,
            transform_sequence.target_transform_space,
        )
    if direction == "inverse":
        return (
            transform_sequence.target_transform_space,
            transform_sequence.source_transform_space,
        )
    raise ValueError(f"Unsupported transform direction: {direction}")


def _reorient_image_to_space(
    image_config: ImageConfig,
    target_space: SpaceDefinition,
    *,
    output_path: Path,
    image_id_suffix: str,
    dtype=None,
    output_image_id_prefix: str | None = None,
) -> ImageConfig:
    input_array = np.asarray(nib.load(str(image_config.image)).dataobj)
    output_array = input_array
    output_space = image_config.space

    if image_config.space.orientation != target_space.orientation:
        output_array, output_space = reorient_array_to_match(
            input_array,
            image_config.space,
            target_space,
        )

    write_nifti_from_array(
        output_array,
        output_space,
        output_path,
        dtype=dtype,
    )
    image_id_prefix = output_image_id_prefix or image_config.image_id
    return ImageConfig(
        image_id=f"{image_id_prefix}_{image_id_suffix}",
        image=output_path,
        space=output_space,
    )


def _derived_output_path(output_path: Path, label: str) -> Path:
    if output_path.suffix == ".gz" and output_path.name.endswith(".nii.gz"):
        stem = output_path.name[:-7]
        return output_path.with_name(f"{stem}_{label}.nii.gz")
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}_{label}{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}_{label}")


def _cleanup_intermediate_paths(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _validate_space_compatibility(
    *,
    actual: SpaceDefinition,
    expected: SpaceDefinition,
    label: str,
) -> None:
    shared_fields = (
        "space_name",
        "orientation",
        "units",
        "resolution_um",
    )
    mismatches = [
        field_name
        for field_name in shared_fields
        if getattr(actual, field_name) != getattr(expected, field_name)
    ]
    if actual.shape is not None and expected.shape is not None and actual.shape != expected.shape:
        mismatches.append("shape")

    if mismatches:
        mismatch_str = ", ".join(mismatches)
        raise ValueError(
            f"{label} is incompatible with the transform space definition. "
            f"Mismatched fields: {mismatch_str}."
        )


def _spaces_are_compatible(
    actual: SpaceDefinition,
    expected: SpaceDefinition,
) -> bool:
    shared_fields = (
        "space_name",
        "orientation",
        "units",
        "resolution_um",
    )
    if any(getattr(actual, field_name) != getattr(expected, field_name) for field_name in shared_fields):
        return False
    if actual.shape is not None and expected.shape is not None and actual.shape != expected.shape:
        return False
    return True
