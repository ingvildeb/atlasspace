from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from atlasbuilder.config.config_models import ImageConfig
from atlasbuilder.config.space_models import SpaceDefinition
from atlasbuilder.image._image_config_utils import (
    build_output_image_config,
    validate_binary_mask_array,
    validate_identical_masking_spaces,
    validate_or_fill_space_shape,
)
from atlasbuilder.io.nifti import write_nifti_from_array
from atlasbuilder.runtime.template import TemplateAccumulationResult


def _load_array_and_space(image_config: ImageConfig) -> tuple[np.ndarray, SpaceDefinition]:
    nifti = nib.load(str(image_config.image))
    array = np.asanyarray(nifti.dataobj, dtype=np.float32)
    space = validate_or_fill_space_shape(
        image_config,
        tuple(int(v) for v in array.shape),
    )
    return array, space


def _validate_sequence_lengths(
    subject_configs: Sequence[ImageConfig],
    weight_configs: Sequence[ImageConfig],
    valid_mask_configs: Sequence[ImageConfig],
    confidence_configs: Sequence[ImageConfig] | None,
) -> None:
    if not subject_configs:
        raise ValueError("subject_configs must not be empty.")
    if len(weight_configs) != len(subject_configs):
        raise ValueError("weight_configs must have the same length as subject_configs.")
    if len(valid_mask_configs) != len(subject_configs):
        raise ValueError("valid_mask_configs must have the same length as subject_configs.")
    if confidence_configs is not None and len(confidence_configs) != len(subject_configs):
        raise ValueError("confidence_configs must have the same length as subject_configs.")


def accumulate_template_inputs(
    subject_configs: Sequence[ImageConfig],
    weight_configs: Sequence[ImageConfig],
    valid_mask_configs: Sequence[ImageConfig],
    *,
    confidence_configs: Sequence[ImageConfig] | None = None,
) -> TemplateAccumulationResult:
    """Accumulate voxelwise sums and support maps for one template-update round."""
    _validate_sequence_lengths(
        subject_configs,
        weight_configs,
        valid_mask_configs,
        confidence_configs,
    )

    reference_config = subject_configs[0]
    reference_array, reference_space = _load_array_and_space(reference_config)
    array_shape = reference_array.shape

    weighted_sum = np.zeros(array_shape, dtype=np.float32)
    weight_sum = np.zeros(array_shape, dtype=np.float32)
    plain_sum = np.zeros(array_shape, dtype=np.float32)
    support_count = np.zeros(array_shape, dtype=np.float32)
    confidence_sum = (
        np.zeros(array_shape, dtype=np.float32) if confidence_configs is not None else None
    )

    for index, subject_config in enumerate(subject_configs):
        subject_array, subject_space = _load_array_and_space(subject_config)
        validate_identical_masking_spaces(reference_space, subject_space)

        weight_array, weight_space = _load_array_and_space(weight_configs[index])
        validate_identical_masking_spaces(reference_space, weight_space)
        if np.any(weight_array < 0):
            raise ValueError("Weight images must not contain negative values.")

        valid_mask_array, valid_mask_space = _load_array_and_space(valid_mask_configs[index])
        validate_identical_masking_spaces(reference_space, valid_mask_space)
        validate_binary_mask_array(valid_mask_array)
        valid_mask_bool = valid_mask_array.astype(bool)

        weighted_sum[valid_mask_bool] += (
            subject_array[valid_mask_bool] * weight_array[valid_mask_bool]
        )
        weight_sum[valid_mask_bool] += weight_array[valid_mask_bool]
        plain_sum[valid_mask_bool] += subject_array[valid_mask_bool]
        support_count[valid_mask_bool] += 1.0

        if confidence_configs is not None and confidence_sum is not None:
            confidence_array, confidence_space = _load_array_and_space(confidence_configs[index])
            validate_identical_masking_spaces(reference_space, confidence_space)
            confidence_sum[valid_mask_bool] += confidence_array[valid_mask_bool]

    return TemplateAccumulationResult(
        reference_config=reference_config,
        weighted_sum=weighted_sum,
        weight_sum=weight_sum,
        plain_sum=plain_sum,
        support_count=support_count,
        confidence_sum=confidence_sum,
        subject_count=len(subject_configs),
    )


def finalize_weighted_average_template(
    accumulation: TemplateAccumulationResult,
    output_path: Path,
) -> ImageConfig:
    """Write the voxelwise weighted average from an accumulated template round."""
    output_array = np.zeros_like(accumulation.weighted_sum, dtype=np.float32)
    nonzero_mask = accumulation.weight_sum > 0
    output_array[nonzero_mask] = (
        accumulation.weighted_sum[nonzero_mask] / accumulation.weight_sum[nonzero_mask]
    )

    output_space = accumulation.reference_config.space.model_copy(
        update={"shape": tuple(int(v) for v in output_array.shape)}
    )
    write_nifti_from_array(output_array, output_space, output_path)
    return build_output_image_config(accumulation.reference_config, output_path, output_space)


def finalize_plain_average_template(
    accumulation: TemplateAccumulationResult,
    output_path: Path,
) -> ImageConfig:
    """Write the voxelwise plain average from an accumulated template round."""
    output_array = np.zeros_like(accumulation.plain_sum, dtype=np.float32)
    nonzero_mask = accumulation.support_count > 0
    output_array[nonzero_mask] = (
        accumulation.plain_sum[nonzero_mask] / accumulation.support_count[nonzero_mask]
    )

    output_space = accumulation.reference_config.space.model_copy(
        update={"shape": tuple(int(v) for v in output_array.shape)}
    )
    write_nifti_from_array(output_array, output_space, output_path)
    return build_output_image_config(accumulation.reference_config, output_path, output_space)


def build_weight_sum_image(
    accumulation: TemplateAccumulationResult,
    output_path: Path,
) -> ImageConfig:
    """Write the summed voxelwise weight image for QC and downstream blending."""
    output_space = accumulation.reference_config.space.model_copy(
        update={"shape": tuple(int(v) for v in accumulation.weight_sum.shape)}
    )
    write_nifti_from_array(accumulation.weight_sum, output_space, output_path)
    return build_output_image_config(accumulation.reference_config, output_path, output_space)


def build_support_count_image(
    accumulation: TemplateAccumulationResult,
    output_path: Path,
) -> ImageConfig:
    """Write the voxelwise count of contributing subjects inside the valid masks."""
    output_space = accumulation.reference_config.space.model_copy(
        update={"shape": tuple(int(v) for v in accumulation.support_count.shape)}
    )
    write_nifti_from_array(accumulation.support_count, output_space, output_path)
    return build_output_image_config(accumulation.reference_config, output_path, output_space)


def build_mean_weight_image(
    accumulation: TemplateAccumulationResult,
    output_path: Path,
) -> ImageConfig:
    """Write the mean voxelwise weight among contributing subjects."""
    output_array = np.zeros_like(accumulation.weight_sum, dtype=np.float32)
    nonzero_mask = accumulation.support_count > 0
    output_array[nonzero_mask] = (
        accumulation.weight_sum[nonzero_mask] / accumulation.support_count[nonzero_mask]
    )

    output_space = accumulation.reference_config.space.model_copy(
        update={"shape": tuple(int(v) for v in output_array.shape)}
    )
    write_nifti_from_array(output_array, output_space, output_path)
    return build_output_image_config(accumulation.reference_config, output_path, output_space)


def build_mean_confidence_image(
    accumulation: TemplateAccumulationResult,
    output_path: Path,
) -> ImageConfig:
    """Write the mean voxelwise confidence among contributing subjects."""
    if accumulation.confidence_sum is None:
        raise ValueError(
            "Mean confidence image requires confidence_sum data in TemplateAccumulationResult."
        )

    output_array = np.zeros_like(accumulation.confidence_sum, dtype=np.float32)
    nonzero_mask = accumulation.support_count > 0
    output_array[nonzero_mask] = (
        accumulation.confidence_sum[nonzero_mask] / accumulation.support_count[nonzero_mask]
    )

    output_space = accumulation.reference_config.space.model_copy(
        update={"shape": tuple(int(v) for v in output_array.shape)}
    )
    write_nifti_from_array(output_array, output_space, output_path)
    return build_output_image_config(accumulation.reference_config, output_path, output_space)


def blend_template_with_new_average(
    current_template_config: ImageConfig,
    new_average_config: ImageConfig,
    support_count_config: ImageConfig,
    output_path: Path,
    *,
    existing_template_subject_count: int,
) -> ImageConfig:
    """Blend the current template with the new-round average before any symmetry step."""
    if existing_template_subject_count <= 0:
        raise ValueError("existing_template_subject_count must be positive.")

    current_array, current_space = _load_array_and_space(current_template_config)
    new_average_array, new_average_space = _load_array_and_space(new_average_config)
    support_count_array, support_count_space = _load_array_and_space(support_count_config)

    validate_identical_masking_spaces(current_space, new_average_space)
    validate_identical_masking_spaces(current_space, support_count_space)

    if np.any(support_count_array < 0):
        raise ValueError("Support count image must not contain negative values.")

    output_array = current_array.copy()
    supported_mask = support_count_array > 0
    output_array[supported_mask] = (
        (current_array[supported_mask] * existing_template_subject_count)
        + (new_average_array[supported_mask] * support_count_array[supported_mask])
    ) / (
        existing_template_subject_count + support_count_array[supported_mask]
    )

    output_space = current_space.model_copy(
        update={"shape": tuple(int(v) for v in output_array.shape)}
    )
    write_nifti_from_array(output_array, output_space, output_path)
    return build_output_image_config(current_template_config, output_path, output_space)
