from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from atlasspace.config.image_models import ImageConfig
from atlasspace.image._image_config_utils import (
    build_output_image_config,
    validate_binary_mask_array,
    validate_identical_masking_spaces,
    validate_or_fill_space_shape,
)
from atlasspace.io.nifti import write_nifti_from_array


def segmentation_to_binary_mask(
    segmentation_config: ImageConfig,
    output_path: Path,
) -> ImageConfig:
    segmentation_nifti = nib.load(str(segmentation_config.image))
    segmentation_data = np.asanyarray(segmentation_nifti.dataobj)
    segmentation_space = validate_or_fill_space_shape(
        segmentation_config,
        tuple(int(v) for v in segmentation_data.shape),
    )

    output_data = (segmentation_data > 0).astype(np.uint8, copy=False)
    output_space = segmentation_space.model_copy(
        update={"shape": tuple(int(v) for v in output_data.shape)}
    )
    write_nifti_from_array(
        output_data,
        output_space,
        output_path,
        dtype=np.uint8,
    )
    return build_output_image_config(segmentation_config, output_path, output_space)


def apply_binary_mask(
    image_config: ImageConfig,
    mask_config: ImageConfig,
    output_path: Path,
    *,
    fill_value: float = 0.0,
) -> ImageConfig:
    image_nifti = nib.load(str(image_config.image))
    mask_nifti = nib.load(str(mask_config.image))

    image_data = np.asanyarray(image_nifti.dataobj)
    mask_data = np.asanyarray(mask_nifti.dataobj)

    image_space = validate_or_fill_space_shape(
        image_config,
        tuple(int(v) for v in image_data.shape),
    )
    mask_space = validate_or_fill_space_shape(
        mask_config,
        tuple(int(v) for v in mask_data.shape),
    )
    validate_identical_masking_spaces(image_space, mask_space)
    validate_binary_mask_array(mask_data)

    output_data = np.where(mask_data == 1, image_data, fill_value)
    output_space = image_space.model_copy(
        update={"shape": tuple(int(v) for v in output_data.shape)}
    )
    write_nifti_from_array(
        output_data,
        output_space,
        output_path,
        dtype=image_nifti.get_data_dtype(),
    )
    return build_output_image_config(image_config, output_path, output_space)
