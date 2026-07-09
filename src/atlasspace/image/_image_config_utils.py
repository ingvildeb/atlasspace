from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from atlasspace.config.image_models import ImageConfig
from atlasspace.config.space_models import SpaceDefinition


InterpolationMode = Literal["nearest", "linear", "cubic"]


def validate_or_fill_space_shape(
    image_config: ImageConfig,
    loaded_shape: tuple[int, int, int],
) -> SpaceDefinition:
    if image_config.space.shape is not None and tuple(image_config.space.shape) != tuple(
        loaded_shape
    ):
        raise ValueError(
            "Loaded image shape does not match ImageConfig.space.shape. "
            f"Configured shape: {image_config.space.shape}; loaded shape: {loaded_shape}."
        )
    return image_config.space.model_copy(update={"shape": tuple(int(v) for v in loaded_shape)})


def build_output_image_config(
    image_config: ImageConfig,
    output_path: Path,
    output_space: SpaceDefinition,
) -> ImageConfig:
    return ImageConfig(
        image_id=image_config.image_id,
        image=output_path,
        space=output_space,
    )


def validate_binary_mask_array(mask_array: np.ndarray) -> None:
    unique_mask_values = np.unique(mask_array)
    if not np.all(np.isin(unique_mask_values, [0, 1])):
        raise ValueError(
            "Mask must be binary with values only in {0, 1}. "
            f"Observed values: {unique_mask_values.tolist()}"
        )


def validate_identical_masking_spaces(
    image_space: SpaceDefinition,
    mask_space: SpaceDefinition,
) -> None:
    if tuple(image_space.shape) != tuple(mask_space.shape):
        raise ValueError(
            "Image and mask must have identical shapes for voxelwise masking. "
            f"Got image shape {image_space.shape} and mask shape {mask_space.shape}."
        )
    if image_space.orientation != mask_space.orientation:
        raise ValueError(
            "Image and mask must have identical orientations for voxelwise masking. "
            f"Got image orientation '{image_space.orientation}' and mask orientation "
            f"'{mask_space.orientation}'."
        )
    if tuple(image_space.resolution_um) != tuple(mask_space.resolution_um):
        raise ValueError(
            "Image and mask must have identical resolutions for voxelwise masking. "
            f"Got image resolution {image_space.resolution_um} and mask resolution "
            f"{mask_space.resolution_um}."
        )


def interpolation_to_order(interpolation: InterpolationMode) -> int:
    interpolation_to_order_map = {
        "nearest": 0,
        "linear": 1,
        "cubic": 3,
    }
    return interpolation_to_order_map[interpolation]


def output_dtype_for_interpolation(
    reference_dtype,
    interpolation: InterpolationMode,
):
    if interpolation == "nearest":
        return reference_dtype
    if np.issubdtype(reference_dtype, np.floating):
        return reference_dtype
    return np.float32
