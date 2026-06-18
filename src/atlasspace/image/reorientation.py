from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from atlasspace.config.config_models import ImageConfig
from atlasspace.config.space_models import SpaceDefinition
from atlasspace.image._image_config_utils import (
    build_output_image_config,
    validate_or_fill_space_shape,
)
from atlasspace.io.nifti import write_nifti_from_array


def _letter_to_family(letter: str) -> str:
    if letter in {"l", "r"}:
        return "lr"
    if letter in {"a", "p"}:
        return "ap"
    return "si"


def spaces_match_orientation(
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
) -> bool:
    return source_space.orientation.lower() == target_space.orientation.lower()


def compute_reorientation_transform(
    source_orientation: str,
    target_orientation: str,
) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    source_letters = list(source_orientation.strip().lower())
    target_letters = list(target_orientation.strip().lower())
    source_families = [_letter_to_family(letter) for letter in source_letters]

    permutation: list[int] = []
    flips: list[bool] = []
    for target_letter in target_letters:
        target_family = _letter_to_family(target_letter)
        source_axis = source_families.index(target_family)
        permutation.append(source_axis)
        flips.append(source_letters[source_axis] != target_letter)

    return tuple(permutation), tuple(flips)


def reorient_array_to_match(
    array: np.ndarray,
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
) -> tuple[np.ndarray, SpaceDefinition]:
    if spaces_match_orientation(source_space, target_space):
        return array, source_space

    permutation, flips = compute_reorientation_transform(
        source_space.orientation,
        target_space.orientation,
    )

    reoriented_array = np.transpose(array, axes=permutation)
    for axis, should_flip in enumerate(flips):
        if should_flip:
            reoriented_array = np.flip(reoriented_array, axis=axis)

    shape = tuple(int(value) for value in reoriented_array.shape)
    reoriented_space = SpaceDefinition(
        space_name=source_space.space_name,
        orientation=target_space.orientation,
        axis_labels=tuple(source_space.axis_labels[source_axis] for source_axis in permutation),
        units=source_space.units,
        resolution_um=tuple(
            float(source_space.resolution_um[source_axis]) for source_axis in permutation
        ),
        shape=shape,
    )
    return reoriented_array, reoriented_space


def reorient_space_to_match(
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
) -> SpaceDefinition:
    if spaces_match_orientation(source_space, target_space):
        return source_space

    permutation, _ = compute_reorientation_transform(
        source_space.orientation,
        target_space.orientation,
    )

    shape: tuple[int, int, int] | None = None
    if source_space.shape is not None:
        shape = tuple(int(source_space.shape[source_axis]) for source_axis in permutation)

    return SpaceDefinition(
        space_name=source_space.space_name,
        orientation=target_space.orientation,
        axis_labels=tuple(source_space.axis_labels[source_axis] for source_axis in permutation),
        units=source_space.units,
        resolution_um=tuple(
            float(source_space.resolution_um[source_axis]) for source_axis in permutation
        ),
        shape=shape,
    )


def reorient_image_to_match(
    image_config: ImageConfig,
    target_space: SpaceDefinition,
    output_path: Path,
) -> ImageConfig:
    image_nifti = nib.load(str(image_config.image))
    input_array = np.asanyarray(image_nifti.dataobj)
    input_space = validate_or_fill_space_shape(
        image_config,
        tuple(int(v) for v in input_array.shape),
    )
    reoriented_array, output_space = reorient_array_to_match(
        input_array,
        input_space,
        target_space,
    )
    write_nifti_from_array(
        reoriented_array,
        output_space,
        output_path,
        dtype=image_nifti.get_data_dtype(),
    )
    return build_output_image_config(image_config, output_path, output_space)
