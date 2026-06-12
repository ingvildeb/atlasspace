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
    splenium: np.ndarray


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
    for name, coords in (
        ("WHS", landmarks.whs),
        ("central canal", landmarks.central_canal),
        ("splenium", landmarks.splenium),
    ):
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


def _normalize_vector(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError(f"{label} must not have zero length.")
    return vector / norm


def _build_landmark_frame(
    landmarks: TemplateLandmarks,
    label: str,
) -> np.ndarray:
    primary_axis = _normalize_vector(
        landmarks.whs - landmarks.central_canal,
        f"{label} WHS-central_canal axis",
    )
    splenium_vector = landmarks.splenium - landmarks.whs
    secondary_seed = splenium_vector - np.dot(splenium_vector, primary_axis) * primary_axis
    secondary_norm = float(np.linalg.norm(secondary_seed))
    if secondary_norm <= 1e-6:
        raise ValueError(
            f"{label} splenium landmark is too close to collinear with the WHS-central canal axis "
            "to define a stable 3D frame."
        )
    secondary_axis = secondary_seed / secondary_norm
    tertiary_axis = _normalize_vector(
        np.cross(primary_axis, secondary_axis),
        f"{label} tertiary landmark frame axis",
    )
    secondary_axis = _normalize_vector(
        np.cross(tertiary_axis, primary_axis),
        f"{label} corrected secondary landmark frame axis",
    )
    return np.column_stack((primary_axis, secondary_axis, tertiary_axis))


def compute_pose_standardization_transform(
    source_landmarks: TemplateLandmarks,
    reference_landmarks: TemplateLandmarks,
) -> tuple[np.ndarray, np.ndarray]:
    source_frame = _build_landmark_frame(source_landmarks, "source")
    reference_frame = _build_landmark_frame(reference_landmarks, "reference")
    combined_rotation = reference_frame @ source_frame.T
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
        splenium=_convert_to_zero_based(
            _validate_landmark_array("source_landmarks.splenium", source_landmarks.splenium),
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
        splenium=_convert_to_zero_based(
            _validate_landmark_array(
                "reference_landmarks.splenium", reference_landmarks.splenium
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
