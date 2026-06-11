from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

from atlasbuilder.config.config_models import ImageConfig
from atlasbuilder.image._image_config_utils import (
    InterpolationMode,
    build_output_image_config,
    interpolation_to_order,
    validate_or_fill_space_shape,
)
from atlasbuilder.io.nifti import write_nifti_from_array


@dataclass
class TemplateLandmarks:
    whs: np.ndarray
    central_canal: np.ndarray


def _validate_landmark_array(name: str, coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {coords.shape}.")
    return coords


def _convert_to_zero_based(coords: np.ndarray, coordinate_base: int) -> np.ndarray:
    if coordinate_base not in (0, 1):
        raise ValueError(f"Unsupported coordinate base '{coordinate_base}'. Use 0 or 1.")
    return coords - coordinate_base


def _validate_landmarks(
    landmarks: TemplateLandmarks,
    shape: tuple[int, int, int],
    label: str,
) -> None:
    for name, coords in (("WHS", landmarks.whs), ("central canal", landmarks.central_canal)):
        if np.any(coords < 0):
            raise ValueError(f"{label} {name} coordinate {tuple(coords)} is negative.")
        if np.any(coords >= np.array(shape)):
            raise ValueError(
                f"{label} {name} coordinate {tuple(coords)} falls outside image bounds {shape}."
            )

    if landmarks.whs[2] <= landmarks.central_canal[2]:
        raise ValueError(
            f"{label} WHS must be more anterior than central canal. "
            f"Observed coronal coordinates: WHS={landmarks.whs[2]}, "
            f"central canal={landmarks.central_canal[2]}."
        )


def _build_yaw_matrix(ds: float, dc: float) -> tuple[np.ndarray, float]:
    yaw_angle = np.arctan2(ds, dc)
    cos_a = np.cos(yaw_angle)
    sin_a = np.sin(yaw_angle)
    rotation = np.array(
        [
            [cos_a, 0.0, -sin_a],
            [0.0, 1.0, 0.0],
            [sin_a, 0.0, cos_a],
        ],
        dtype=np.float64,
    )
    return rotation, yaw_angle


def _build_pitch_matrix(
    current_dh: float,
    current_dc: float,
    reference_dh: float,
    reference_dc: float,
) -> tuple[np.ndarray, float, float, float]:
    current_angle = np.arctan2(current_dh, current_dc)
    reference_angle = np.arctan2(reference_dh, reference_dc)
    pitch_correction = reference_angle - current_angle

    cos_a = np.cos(pitch_correction)
    sin_a = np.sin(pitch_correction)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ],
        dtype=np.float64,
    )
    return rotation, current_angle, reference_angle, pitch_correction


def compute_pose_standardization_transform(
    source_landmarks: TemplateLandmarks,
    reference_landmarks: TemplateLandmarks,
) -> tuple[np.ndarray, np.ndarray]:
    template_vector = source_landmarks.whs - source_landmarks.central_canal
    reference_vector = reference_landmarks.whs - reference_landmarks.central_canal

    yaw_rotation, _ = _build_yaw_matrix(template_vector[0], template_vector[2])
    template_vector_after_yaw = yaw_rotation @ template_vector
    reference_vector_after_yaw = reference_vector.copy()
    reference_vector_after_yaw[2] = np.hypot(reference_vector[0], reference_vector[2])

    pitch_rotation, _, _, _ = _build_pitch_matrix(
        current_dh=template_vector_after_yaw[1],
        current_dc=template_vector_after_yaw[2],
        reference_dh=reference_vector_after_yaw[1],
        reference_dc=reference_vector_after_yaw[2],
    )

    combined_rotation = pitch_rotation @ yaw_rotation
    translation = reference_landmarks.whs - source_landmarks.whs
    return combined_rotation, translation


def _apply_rigid_transform(
    volume: np.ndarray,
    rotation: np.ndarray,
    pivot: np.ndarray,
    translation: np.ndarray,
    order: int,
    cval: float,
) -> np.ndarray:
    inverse_rotation = rotation.T
    offset = pivot - inverse_rotation @ (pivot + translation)

    transformed = affine_transform(
        volume,
        matrix=inverse_rotation,
        offset=offset,
        output_shape=volume.shape,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=(order > 1),
    )
    return transformed


def _cast_like_input(volume: np.ndarray, reference_dtype) -> np.ndarray:
    if np.issubdtype(reference_dtype, np.integer):
        info = np.iinfo(reference_dtype)
        volume = np.rint(volume)
        volume = np.clip(volume, info.min, info.max)
        return volume.astype(reference_dtype)
    return volume.astype(reference_dtype)


def standardize_image_pose(
    image_config: ImageConfig,
    output_path: Path,
    source_landmarks: TemplateLandmarks,
    reference_landmarks: TemplateLandmarks,
    *,
    coordinate_base: int = 1,
    interpolation: InterpolationMode = "linear",
    fill_value: float = 0.0,
) -> ImageConfig:
    source_landmarks = TemplateLandmarks(
        whs=_convert_to_zero_based(
            _validate_landmark_array("source_landmarks.whs", source_landmarks.whs),
            coordinate_base,
        ),
        central_canal=_convert_to_zero_based(
            _validate_landmark_array(
                "source_landmarks.central_canal", source_landmarks.central_canal
            ),
            coordinate_base,
        ),
    )
    reference_landmarks = TemplateLandmarks(
        whs=_convert_to_zero_based(
            _validate_landmark_array("reference_landmarks.whs", reference_landmarks.whs),
            coordinate_base,
        ),
        central_canal=_convert_to_zero_based(
            _validate_landmark_array(
                "reference_landmarks.central_canal", reference_landmarks.central_canal
            ),
            coordinate_base,
        ),
    )

    image_nifti = nib.load(str(image_config.image))
    input_data = np.asarray(image_nifti.dataobj, dtype=np.float32)
    input_dtype = image_nifti.get_data_dtype()
    input_space = validate_or_fill_space_shape(
        image_config,
        tuple(int(v) for v in input_data.shape),
    )

    _validate_landmarks(source_landmarks, input_data.shape, "source")
    _validate_landmarks(reference_landmarks, input_data.shape, "reference")

    rotation, translation = compute_pose_standardization_transform(
        source_landmarks,
        reference_landmarks,
    )

    transformed = _apply_rigid_transform(
        volume=input_data,
        rotation=rotation,
        pivot=source_landmarks.whs,
        translation=translation,
        order=interpolation_to_order(interpolation),
        cval=fill_value,
    )

    output_data = _cast_like_input(transformed, input_dtype)
    output_space = input_space.model_copy(
        update={"shape": tuple(int(v) for v in output_data.shape)}
    )
    write_nifti_from_array(
        output_data,
        output_space,
        output_path,
        dtype=input_dtype,
    )
    return build_output_image_config(image_config, output_path, output_space)
