from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from atlasspace.config.config_models import ImageConfig
from atlasspace.image._image_config_utils import (
    InterpolationMode,
    build_output_image_config,
    interpolation_to_order,
    output_dtype_for_interpolation,
    validate_or_fill_space_shape,
)
from atlasspace.io.nifti import write_nifti_from_array


def resample_image_to_resolution(
    image_config: ImageConfig,
    output_path: Path,
    *,
    target_resolution_um: tuple[float, float, float],
    interpolation: InterpolationMode = "linear",
) -> ImageConfig:
    image_nifti = nib.load(str(image_config.image))
    input_array = np.asanyarray(image_nifti.dataobj)
    input_space = validate_or_fill_space_shape(
        image_config,
        tuple(int(v) for v in input_array.shape),
    )

    zoom_factors = tuple(
        float(source_resolution) / float(target_resolution)
        for source_resolution, target_resolution in zip(
            input_space.resolution_um,
            target_resolution_um,
            strict=True,
        )
    )

    if all(factor == 1.0 for factor in zoom_factors):
        resampled_array = input_array.copy()
    else:
        resampled_array = zoom(
            input_array,
            zoom=zoom_factors,
            order=interpolation_to_order(interpolation),
            mode="constant",
            cval=0.0,
            prefilter=(interpolation == "cubic"),
        )

    output_space = input_space.model_copy(
        update={
            "resolution_um": tuple(float(v) for v in target_resolution_um),
            "shape": tuple(int(v) for v in resampled_array.shape),
        }
    )
    write_nifti_from_array(
        resampled_array,
        output_space,
        output_path,
        dtype=output_dtype_for_interpolation(image_nifti.get_data_dtype(), interpolation),
    )
    return build_output_image_config(image_config, output_path, output_space)
