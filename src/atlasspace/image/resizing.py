from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from atlasspace.config.config_models import ImageConfig
from atlasspace.image._image_config_utils import (
    build_output_image_config,
    validate_or_fill_space_shape,
)
from atlasspace.io.nifti import write_nifti_from_array


def _center_crop_and_pad(
    array: np.ndarray,
    target_shape: tuple[int, int, int],
    fill_value: float,
) -> np.ndarray:
    resized = array

    crop_slices = []
    pad_width = []
    for current_size, target_size in zip(array.shape, target_shape, strict=True):
        if current_size > target_size:
            excess = current_size - target_size
            crop_before = excess // 2
            crop_after = excess - crop_before
            crop_slices.append(slice(crop_before, current_size - crop_after))
            pad_width.append((0, 0))
        else:
            crop_slices.append(slice(0, current_size))
            deficit = target_size - current_size
            pad_before = deficit // 2
            pad_after = deficit - pad_before
            pad_width.append((pad_before, pad_after))

    resized = resized[tuple(crop_slices)]
    if any(before > 0 or after > 0 for before, after in pad_width):
        resized = np.pad(
            resized,
            pad_width,
            mode="constant",
            constant_values=fill_value,
        )
    return resized


def resize_image_to_shape(
    image_config: ImageConfig,
    output_path: Path,
    *,
    target_shape: tuple[int, int, int],
    fill_value: float = 0.0,
) -> ImageConfig:
    image_nifti = nib.load(str(image_config.image))
    input_array = np.asanyarray(image_nifti.dataobj)
    input_space = validate_or_fill_space_shape(
        image_config,
        tuple(int(v) for v in input_array.shape),
    )

    resized_array = _center_crop_and_pad(input_array, target_shape, fill_value)
    output_space = input_space.model_copy(
        update={"shape": tuple(int(v) for v in resized_array.shape)}
    )
    write_nifti_from_array(
        resized_array,
        output_space,
        output_path,
        dtype=image_nifti.get_data_dtype(),
    )
    return build_output_image_config(image_config, output_path, output_space)
