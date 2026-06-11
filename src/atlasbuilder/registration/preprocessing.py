from __future__ import annotations

import numpy as np

from atlasbuilder.config.config_models import PreprocessingConfig

try:
    import ants
except ImportError as exc:
    raise ImportError(
        "atlasbuilder.registration.preprocess requires antspyx. "
        "Install atlasbuilder with the registration dependencies to use this module."
    ) from exc


def _numpy_from_ants(image: ants.ANTsImage) -> np.ndarray:
    return image.numpy().astype(np.float32, copy=False)


def apply_intensity_normalization(
    image: ants.ANTsImage,
    mode: str | None,
) -> ants.ANTsImage:
    if mode is None:
        return image

    data = _numpy_from_ants(image).copy()

    if mode == "zscore":
        mean = float(data.mean())
        std = float(data.std())
        if std > 0:
            data = (data - mean) / std
    elif mode == "robust_zscore":
        median = float(np.median(data))
        mad = float(np.median(np.abs(data - median)))
        scale = 1.4826 * mad
        if scale > 0:
            data = (data - median) / scale
    else:
        raise ValueError(f"Unsupported intensity normalization mode: {mode}")

    return ants.from_numpy(
        data.astype(np.float32),
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
    )


def clip_and_rescale_intensity(
    image: ants.ANTsImage,
    percentiles: tuple[float, float] | None,
) -> ants.ANTsImage:
    if percentiles is None:
        return image

    low, high = percentiles
    data = _numpy_from_ants(image).copy()
    low_value = float(np.percentile(data, low))
    high_value = float(np.percentile(data, high))
    data = np.clip(data, low_value, high_value)
    if high_value > low_value:
        data = (data - low_value) / (high_value - low_value)

    return ants.from_numpy(
        data.astype(np.float32),
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
    )


def histogram_match_image(
    moving: ants.ANTsImage,
    fixed: ants.ANTsImage,
    enabled: bool = True,
) -> ants.ANTsImage:
    if not enabled:
        return moving
    return ants.histogram_match_image(moving, fixed)


def smooth_image(
    image: ants.ANTsImage,
    sigma_vox: float | None,
) -> ants.ANTsImage:
    if sigma_vox is None or sigma_vox <= 0:
        return image
    return ants.smooth_image(image, sigma=sigma_vox, sigma_in_physical_coordinates=False)


def resample_to_resolution(
    image: ants.ANTsImage,
    nominal_resolution_um: float,
    target_resolution_um: float,
) -> ants.ANTsImage:
    if target_resolution_um == nominal_resolution_um:
        return image
    factor = float(target_resolution_um) / float(nominal_resolution_um)
    new_spacing = tuple(float(value) * factor for value in image.spacing)
    return ants.resample_image(image, new_spacing, use_voxels=False, interp_type=0)


def preprocess_registration_images(
    fixed_image: ants.ANTsImage,
    moving_image: ants.ANTsImage,
    preprocessing_config: PreprocessingConfig,
) -> tuple[ants.ANTsImage, ants.ANTsImage]:
    fixed_preprocessed = fixed_image.clone()
    moving_preprocessed = moving_image.clone()

    fixed_preprocessed = apply_intensity_normalization(
        fixed_preprocessed,
        preprocessing_config.intensity_normalization,
    )
    moving_preprocessed = apply_intensity_normalization(
        moving_preprocessed,
        preprocessing_config.intensity_normalization,
    )

    fixed_preprocessed = clip_and_rescale_intensity(
        fixed_preprocessed,
        preprocessing_config.minmax_clip_percentiles,
    )
    moving_preprocessed = clip_and_rescale_intensity(
        moving_preprocessed,
        preprocessing_config.minmax_clip_percentiles,
    )

    moving_preprocessed = histogram_match_image(
        moving_preprocessed,
        fixed_preprocessed,
        enabled=preprocessing_config.histogram_match,
    )

    fixed_preprocessed = smooth_image(
        fixed_preprocessed,
        preprocessing_config.gaussian_sigma_vox,
    )
    moving_preprocessed = smooth_image(
        moving_preprocessed,
        preprocessing_config.gaussian_sigma_vox,
    )

    return fixed_preprocessed, moving_preprocessed
