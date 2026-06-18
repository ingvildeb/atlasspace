from __future__ import annotations

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from atlasspace.config.config_models import ImageConfig
from atlasspace.config.space_models import SpaceDefinition
from atlasspace.image._image_config_utils import (
    build_output_image_config,
    validate_binary_mask_array,
    validate_identical_masking_spaces,
    validate_or_fill_space_shape,
)
from atlasspace.io.nifti import write_nifti_from_array


CONFIDENCE_MIN_VALUE = 0.0
CONFIDENCE_MAX_VALUE = 1.0


def _load_array_and_space(image_config: ImageConfig) -> tuple[np.ndarray, SpaceDefinition]:
    """ Load array and space metadata based on ImageConfig instance """
    nifti = nib.load(str(image_config.image))
    array = np.asanyarray(nifti.dataobj, dtype=np.float32)
    space = validate_or_fill_space_shape(
        image_config,
        tuple(int(v) for v in array.shape),
    )
    return array, space


def _normalize_array_in_mask(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    *,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, float, float]:
    """  Normalize an array within a mask and return upper and lower bound values """
    masked_values = image_array[mask_array]
    masked_values = masked_values[np.isfinite(masked_values)]

    if masked_values.size == 0:
        raise ValueError("Cannot normalize an image with an empty masked region.")

    lower_bound = float(np.percentile(masked_values, lower_percentile))
    upper_bound = float(np.percentile(masked_values, upper_percentile))

    if upper_bound <= lower_bound:
        return np.zeros_like(image_array, dtype=np.float32), lower_bound, upper_bound

    normalized_array = (image_array - lower_bound) / (upper_bound - lower_bound)
    normalized_array = np.clip(normalized_array, 0.0, 1.0).astype(np.float32)
    return normalized_array, lower_bound, upper_bound


def _normalize_array_with_reference_bounds(
    image_array: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """ Apply a set of normalization bound values to an array """
    if upper_bound <= lower_bound:
        return np.zeros_like(image_array, dtype=np.float32)

    normalized_array = (image_array - lower_bound) / (upper_bound - lower_bound)
    return np.clip(normalized_array, 0.0, 1.0).astype(np.float32)


def _histogram_match_array_in_mask(
    source_array: np.ndarray,
    reference_array: np.ndarray,
    mask_array: np.ndarray,
) -> np.ndarray:
    """ Histogram match one array to another within a mask """
    matched_array = source_array.copy().astype(np.float32)
    source_values = source_array[mask_array]
    reference_values = reference_array[mask_array]

    source_values = source_values[np.isfinite(source_values)]
    reference_values = reference_values[np.isfinite(reference_values)]

    if source_values.size == 0 or reference_values.size == 0:
        raise ValueError(
            "Cannot histogram-match because masked source/reference values are empty."
        )

    source_values_sorted = np.sort(source_values)
    reference_values_sorted = np.sort(reference_values)
    source_quantiles = np.linspace(0.0, 1.0, source_values_sorted.size)
    reference_quantiles = np.linspace(0.0, 1.0, reference_values_sorted.size)
    source_value_quantiles = np.interp(
        source_values,
        source_values_sorted,
        source_quantiles,
    )
    matched_values = np.interp(
        source_value_quantiles,
        reference_quantiles,
        reference_values_sorted,
    )
    matched_array[mask_array] = matched_values.astype(np.float32)
    return matched_array


def _smooth_array(image_array: np.ndarray, sigma_voxels: float) -> np.ndarray:
    """ Gaussian filter smoothing of an array """
    if sigma_voxels <= 0:
        return image_array.astype(np.float32, copy=False)
    return gaussian_filter(image_array, sigma=sigma_voxels).astype(np.float32)


def _compute_residual_array(
    subject_array: np.ndarray,
    template_array: np.ndarray,
    *,
    residual_mode: Literal["absolute", "relative"],
    template_relative_floor: float,
) -> np.ndarray:
    """ Compute residual differences between two arrays using either 
        absolute or relative differences """
    absolute_difference = np.abs(subject_array - template_array)

    if residual_mode == "absolute":
        return absolute_difference.astype(np.float32)
    if residual_mode == "relative":
        denominator = template_array + template_relative_floor
        return (absolute_difference / denominator).astype(np.float32)

    raise ValueError(f"Unsupported residual_mode: {residual_mode}")


def build_confidence_map(
    subject_config: ImageConfig,
    template_config: ImageConfig,
    valid_mask_config: ImageConfig,
    output_path: Path,
    *,
    histogram_match: bool = True,
    smoothing_sigma_voxels: float = 1.5,
    output_smoothing_sigma_voxels: float = 0.0,
    residual_mode: Literal["absolute", "relative"] = "relative",
    template_relative_floor: float = 0.10,
    normalization_lower_percentile: float = 1.0,
    normalization_upper_percentile: float = 99.0,
    residual_low_percentile: float = 5.0,
    residual_high_percentile: float = 99.0,
) -> ImageConfig:
    """Build a voxelwise confidence map for one registered subject 
        in template space."""
    if smoothing_sigma_voxels < 0 or output_smoothing_sigma_voxels < 0:
        raise ValueError("Smoothing sigmas must be nonnegative.")
    if template_relative_floor <= 0:
        raise ValueError("template_relative_floor must be positive.")

    # Load arrays and space metadata for subject, template and mask
    subject_array, subject_space = _load_array_and_space(subject_config)
    template_array, template_space = _load_array_and_space(template_config)
    valid_mask_array, valid_mask_space = _load_array_and_space(valid_mask_config)

    # Validate spaces match and mask is binary
    validate_identical_masking_spaces(subject_space, template_space)
    validate_identical_masking_spaces(subject_space, valid_mask_space)
    validate_binary_mask_array(valid_mask_array)
    valid_mask_bool = valid_mask_array.astype(bool)

    # Convery to arrays
    subject_array = np.nan_to_num(subject_array, nan=0.0, posinf=0.0, neginf=0.0)
    template_array = np.nan_to_num(template_array, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize template array within mask and return bounds
    normalized_template_array, template_lower_bound, template_upper_bound = (
        _normalize_array_in_mask(
            template_array,
            valid_mask_bool,
            lower_percentile=normalization_lower_percentile,
            upper_percentile=normalization_upper_percentile,
        )
    )

    # Histogram match the subject array to the template array
    if histogram_match:
        matched_subject_array = _histogram_match_array_in_mask(
            subject_array,
            template_array,
            valid_mask_bool,
        )
    else:
        matched_subject_array = subject_array.copy()

    # Normalize the subject array using the template-derived bounds
    normalized_subject_array = _normalize_array_with_reference_bounds(
        matched_subject_array,
        template_lower_bound,
        template_upper_bound,
    )

    # Smooth subject and template arrays
    smoothed_subject_array = _smooth_array(
        normalized_subject_array,
        smoothing_sigma_voxels,
    )
    smoothed_template_array = _smooth_array(
        normalized_template_array,
        smoothing_sigma_voxels,
    )

    # Compute the residual array for subject and template
    residual_array = _compute_residual_array(
        smoothed_subject_array,
        smoothed_template_array,
        residual_mode=residual_mode,
        template_relative_floor=template_relative_floor,
    )
    masked_residual_values = residual_array[valid_mask_bool]

    residual_low_bound = float(
        np.percentile(masked_residual_values, residual_low_percentile)
    )
    residual_high_bound = float(
        np.percentile(masked_residual_values, residual_high_percentile)
    )

    # Scale residual array to values between 0 and 1
    # Compute confidence array by 1 - scaled residual
    if residual_high_bound <= residual_low_bound:
        confidence_array = np.ones_like(residual_array, dtype=np.float32)
    else:
        scaled_residual_array = (
            (residual_array - residual_low_bound)
            / (residual_high_bound - residual_low_bound)
        )
        scaled_residual_array = np.clip(scaled_residual_array, 0.0, 1.0).astype(np.float32)
        confidence_array = 1.0 - scaled_residual_array
        confidence_array = np.clip(
            confidence_array,
            CONFIDENCE_MIN_VALUE,
            CONFIDENCE_MAX_VALUE,
        ).astype(np.float32)

    # Mask the final confidence array
    confidence_array = confidence_array * valid_mask_bool.astype(np.float32)

    # Smooth confidence array
    if output_smoothing_sigma_voxels > 0:
        confidence_array = _smooth_array(confidence_array, output_smoothing_sigma_voxels)
        confidence_array = np.clip(
            confidence_array,
            CONFIDENCE_MIN_VALUE,
            CONFIDENCE_MAX_VALUE,
        ).astype(np.float32)
        confidence_array = confidence_array * valid_mask_bool.astype(np.float32)

    output_space = subject_space.model_copy(
        update={"shape": tuple(int(v) for v in confidence_array.shape)}
    )

    # Write confidence array nifti
    write_nifti_from_array(confidence_array, output_space, output_path)

    # Return ImageConfig objects
    return build_output_image_config(subject_config, output_path, output_space)


def confidence_to_weight_map(
    confidence_config: ImageConfig,
    output_path: Path,
    *,
    confidence_cutoff: float = 0.85,
    confidence_power: float = 2.0,
) -> ImageConfig:
    """Convert a confidence image into a weighting image for template averaging."""
    if not (0.0 <= confidence_cutoff < 1.0):
        raise ValueError("confidence_cutoff must satisfy 0 <= cutoff < 1.")
    if confidence_power <= 0:
        raise ValueError("confidence_power must be positive.")

    confidence_array, confidence_space = _load_array_and_space(confidence_config)

    # Apply cutoff so that only voxels with confidence > cutoff gets weight
    shifted_confidence = (confidence_array - confidence_cutoff) / (1.0 - confidence_cutoff)
    shifted_confidence = np.clip(shifted_confidence, 0.0, 1.0)

    # Apply power formula to amplify higher weights more
    weight_array = np.power(shifted_confidence, confidence_power).astype(np.float32)

    output_space = confidence_space.model_copy(
        update={"shape": tuple(int(v) for v in weight_array.shape)}
    )

    # Write weight array nifti
    write_nifti_from_array(weight_array, output_space, output_path)

    # Return ImageConfig objects
    return build_output_image_config(confidence_config, output_path, output_space)
