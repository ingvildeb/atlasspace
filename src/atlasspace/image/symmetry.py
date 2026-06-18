from __future__ import annotations

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np

from atlasspace.config.config_models import ImageConfig
from atlasspace.image._image_config_utils import (
    build_output_image_config,
    validate_identical_masking_spaces,
    validate_binary_mask_array,
    validate_or_fill_space_shape,
)
from atlasspace.io.nifti import write_nifti_from_array


def _infer_left_right_axis(orientation: str) -> tuple[int, bool]:
    for axis, letter in enumerate(orientation):
        if letter == "l":
            return axis, True
        if letter == "r":
            return axis, False
    raise ValueError(f"Could not infer a left-right axis from orientation '{orientation}'.")


def _default_midline_index(axis_length: int) -> int:
    return (axis_length - 1) // 2


def _reflect_array_lr(array: np.ndarray, orientation: str) -> np.ndarray:
    lr_axis, _ = _infer_left_right_axis(orientation)
    return np.flip(array, axis=lr_axis)


def mirror_unilateral_mask(
    mask_config: ImageConfig,
    output_path: Path,
    *,
    edited_side: Literal["left", "right"],
    midline_index: int | None = None,
) -> ImageConfig:
    mask_nifti = nib.load(str(mask_config.image))
    mask_array = np.asanyarray(mask_nifti.dataobj)
    mask_space = validate_or_fill_space_shape(
        mask_config,
        tuple(int(v) for v in mask_array.shape),
    )
    validate_binary_mask_array(mask_array)

    lr_axis, low_indices_are_left = _infer_left_right_axis(mask_space.orientation)
    axis_length = mask_array.shape[lr_axis]
    if midline_index is None:
        midline_index = _default_midline_index(axis_length)
    if not (0 <= midline_index < axis_length):
        raise ValueError(
            f"midline_index must fall within axis bounds [0, {axis_length - 1}], "
            f"got {midline_index}."
        )

    axis_has_center_voxel = axis_length % 2 == 1
    source_is_low_side = (edited_side == "left" and low_indices_are_left) or (
        edited_side == "right" and not low_indices_are_left
    )

    if source_is_low_side:
        source_start = 0
        source_end = midline_index if axis_has_center_voxel else midline_index + 1
    else:
        source_start = midline_index + 1
        source_end = axis_length

    source_indices = np.arange(source_start, source_end, dtype=int)
    mirror_plane = float(midline_index) if axis_has_center_voxel else float(midline_index) + 0.5
    mirrored_indices = np.rint((2.0 * mirror_plane) - source_indices).astype(int)
    valid = (mirrored_indices >= 0) & (mirrored_indices < axis_length)

    output_array = np.zeros_like(mask_array)

    if source_indices.size > 0:
        source_data = np.take(mask_array, source_indices, axis=lr_axis)

        source_target = [slice(None)] * mask_array.ndim
        source_target[lr_axis] = source_indices
        output_array[tuple(source_target)] = source_data

        mirrored_target = [slice(None)] * mask_array.ndim
        mirrored_target[lr_axis] = mirrored_indices[valid]
        output_array[tuple(mirrored_target)] = np.take(
            source_data,
            np.flatnonzero(valid),
            axis=lr_axis,
        )

    if axis_has_center_voxel:
        center_target = [slice(None)] * mask_array.ndim
        center_target[lr_axis] = midline_index
        output_array[tuple(center_target)] = np.take(mask_array, midline_index, axis=lr_axis)

    output_space = mask_space.model_copy(
        update={"shape": tuple(int(v) for v in output_array.shape)}
    )
    write_nifti_from_array(
        output_array,
        output_space,
        output_path,
        dtype=mask_nifti.get_data_dtype(),
    )
    return build_output_image_config(mask_config, output_path, output_space)


def symmetrize_image_using_support(
    image_config: ImageConfig,
    support_config: ImageConfig,
    output_path: Path,
) -> ImageConfig:
    """Make an image LR-symmetric using voxelwise support to merge reflected pairs."""
    image_nifti = nib.load(str(image_config.image))
    support_nifti = nib.load(str(support_config.image))

    image_array = np.asanyarray(image_nifti.dataobj, dtype=np.float32)
    support_array = np.asanyarray(support_nifti.dataobj, dtype=np.float32)

    image_space = validate_or_fill_space_shape(
        image_config,
        tuple(int(v) for v in image_array.shape),
    )
    support_space = validate_or_fill_space_shape(
        support_config,
        tuple(int(v) for v in support_array.shape),
    )
    validate_identical_masking_spaces(image_space, support_space)

    if np.any(support_array < 0):
        raise ValueError("Support image must not contain negative values.")

    reflected_image_array = _reflect_array_lr(image_array, image_space.orientation)
    reflected_support_array = _reflect_array_lr(support_array, image_space.orientation)

    combined_support_array = support_array + reflected_support_array
    output_array = np.zeros_like(image_array, dtype=np.float32)
    supported_mask = combined_support_array > 0

    combined_value_array = (
        (image_array * support_array)
        + (reflected_image_array * reflected_support_array)
    )
    output_array[supported_mask] = (
        combined_value_array[supported_mask] / combined_support_array[supported_mask]
    )

    output_space = image_space.model_copy(
        update={"shape": tuple(int(v) for v in output_array.shape)}
    )
    write_nifti_from_array(output_array, output_space, output_path)
    return build_output_image_config(image_config, output_path, output_space)


def symmetrize_support_image(
    support_config: ImageConfig,
    output_path: Path,
) -> ImageConfig:
    """Make a support image LR-symmetric by taking the voxelwise max across reflected pairs."""
    support_nifti = nib.load(str(support_config.image))
    support_array = np.asanyarray(support_nifti.dataobj, dtype=np.float32)
    support_space = validate_or_fill_space_shape(
        support_config,
        tuple(int(v) for v in support_array.shape),
    )

    if np.any(support_array < 0):
        raise ValueError("Support image must not contain negative values.")

    reflected_support_array = _reflect_array_lr(support_array, support_space.orientation)
    output_array = np.maximum(support_array, reflected_support_array).astype(np.float32)

    output_space = support_space.model_copy(
        update={"shape": tuple(int(v) for v in output_array.shape)}
    )
    write_nifti_from_array(output_array, output_space, output_path)
    return build_output_image_config(support_config, output_path, output_space)
