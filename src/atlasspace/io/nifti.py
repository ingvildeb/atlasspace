from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError(
        "atlasspace.io.nifti requires nibabel. "
        "Install atlasspace with `pip install -e .` or otherwise ensure nibabel is available."
    ) from exc

from atlasspace.config.space_models import SpaceDefinition


_UM_PER_MM = 1000.0
_BRAINGLOBE_ORIGIN_TO_RAS = {
    "l": (0, 1.0),
    "r": (0, -1.0),
    "p": (1, 1.0),
    "a": (1, -1.0),
    "i": (2, 1.0),
    "s": (2, -1.0),
}
_VALID_ORIENTATION_LETTERS = frozenset(_BRAINGLOBE_ORIGIN_TO_RAS)
_ORIENTATION_FAMILIES = {
    "l": "lr",
    "r": "lr",
    "a": "ap",
    "p": "ap",
    "s": "si",
    "i": "si",
}


def _normalize_declared_orientation(orientation: str) -> str:
    normalized = orientation.strip().lower()
    if len(normalized) != 3:
        raise ValueError("orientation must have length 3.")
    if any(letter not in _VALID_ORIENTATION_LETTERS for letter in normalized):
        raise ValueError("orientation must use letters from {l, r, a, p, s, i}.")

    families = {_ORIENTATION_FAMILIES[letter] for letter in normalized}
    if families != {"lr", "ap", "si"}:
        raise ValueError(
            "orientation must include one left/right, one anterior/posterior, "
            "and one superior/inferior axis."
        )
    return normalized


def _validate_shape(shape: Iterable[int]) -> tuple[int, int, int]:
    shape_tuple = tuple(int(value) for value in shape)
    if len(shape_tuple) != 3:
        raise ValueError(f"shape must have length 3. Got {shape_tuple!r}.")
    if any(value <= 0 for value in shape_tuple):
        raise ValueError(f"shape values must all be positive. Got {shape_tuple!r}.")
    return shape_tuple


def _coerce_isotropic_resolution_um(
    resolution_um: float | int | tuple[float, float, float] | list[float],
) -> float:
    if isinstance(resolution_um, (int, float)):
        resolution_value = float(resolution_um)
        if resolution_value <= 0:
            raise ValueError("resolution_um must be positive.")
        return resolution_value

    resolution_tuple = tuple(float(value) for value in resolution_um)
    if len(resolution_tuple) != 3:
        raise ValueError(
            "resolution_um must be a positive scalar or a length-3 isotropic tuple."
        )
    if any(value <= 0 for value in resolution_tuple):
        raise ValueError("resolution_um values must all be positive.")
    if not np.allclose(resolution_tuple, resolution_tuple[0]):
        raise ValueError(
            "Non-isotropic resolution is not supported yet. "
            f"Got resolution_um={resolution_tuple!r}."
        )
    return resolution_tuple[0]


def _build_nifti_image_with_declared_space(
    array: np.ndarray,
    orientation: str,
    resolution_um: float | int | tuple[float, float, float] | list[float],
    *,
    dtype=None,
    qform_code: int = 1,
    sform_code: int = 1,
) -> nib.Nifti1Image:
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array. Got shape {array.shape!r}.")

    affine = build_nifti_affine_from_declared_space(
        shape=array.shape,
        orientation=orientation,
        resolution_um=resolution_um,
    )

    if dtype is None:
        array_to_write = np.asarray(array)
    else:
        array_to_write = np.asarray(array, dtype=dtype)

    image = nib.Nifti1Image(array_to_write, affine)
    image.header.set_xyzt_units("mm")
    resolution_mm = _coerce_isotropic_resolution_um(resolution_um) / _UM_PER_MM
    image.header.set_zooms((resolution_mm, resolution_mm, resolution_mm))
    image.set_qform(affine, code=qform_code)
    image.set_sform(affine, code=sform_code)
    return image


def build_nifti_affine_from_space(space: SpaceDefinition) -> np.ndarray:
    return build_nifti_affine_from_declared_space(
        shape=space.shape or (1, 1, 1),
        orientation=space.orientation,
        resolution_um=space.resolution_um,
    )


def build_nifti_affine_from_declared_space(
    shape: tuple[int, int, int] | list[int],
    orientation: str,
    resolution_um: float | int | tuple[float, float, float] | list[float],
) -> np.ndarray:
    _validate_shape(shape)
    normalized_orientation = _normalize_declared_orientation(orientation)
    isotropic_resolution_um = _coerce_isotropic_resolution_um(resolution_um)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = 0.0

    resolution_mm = isotropic_resolution_um / _UM_PER_MM
    for axis_index, orientation_letter in enumerate(normalized_orientation):
        world_axis, sign = _BRAINGLOBE_ORIGIN_TO_RAS[orientation_letter]
        affine[world_axis, axis_index] = sign * resolution_mm

    return affine


def rewrite_nifti_header_to_declared_space(
    input_path: Path,
    output_path: Path,
    orientation: str,
    resolution_um: float | int | tuple[float, float, float] | list[float],
    *,
    preserve_data_dtype: bool = True,
    qform_code: int = 1,
    sform_code: int = 1,
) -> Path:
    input_image = nib.load(str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_array = np.asanyarray(input_image.dataobj)
    output_dtype = input_image.get_data_dtype() if preserve_data_dtype else None
    output_image = _build_nifti_image_with_declared_space(
        input_array,
        orientation,
        resolution_um,
        dtype=output_dtype,
        qform_code=qform_code,
        sform_code=sform_code,
    )
    nib.save(output_image, str(output_path))
    return output_path


def save_array_as_nifti_in_declared_space(
    array: np.ndarray,
    output_path: Path,
    orientation: str,
    resolution_um: float | int | tuple[float, float, float] | list[float],
    *,
    dtype=None,
    qform_code: int = 1,
    sform_code: int = 1,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = _build_nifti_image_with_declared_space(
        array,
        orientation,
        resolution_um,
        dtype=dtype,
        qform_code=qform_code,
        sform_code=sform_code,
    )
    nib.save(image, str(output_path))
    return output_path


def write_nifti_from_array(
    array: np.ndarray,
    space: SpaceDefinition,
    output_path: Path,
    *,
    dtype=None,
) -> Path:
    if dtype is None:
        dtype = np.float32
    return save_array_as_nifti_in_declared_space(
        array=array,
        output_path=output_path,
        orientation=space.orientation,
        resolution_um=space.resolution_um,
        dtype=dtype,
    )


def load_nifti_array(input_path: Path) -> np.ndarray:
    image = nib.load(str(input_path))
    return np.asanyarray(image.dataobj)
