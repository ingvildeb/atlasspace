from __future__ import annotations

from pathlib import Path

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


def build_nifti_affine_from_space(space: SpaceDefinition) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = 0.0

    for axis_index, (orientation_letter, resolution_um) in enumerate(
        zip(space.orientation.lower(), space.resolution_um, strict=True)
    ):
        world_axis, sign = _BRAINGLOBE_ORIGIN_TO_RAS[orientation_letter]
        resolution_mm = float(resolution_um) / _UM_PER_MM
        affine[world_axis, axis_index] = sign * resolution_mm

    return affine


def write_nifti_from_array(
    array: np.ndarray,
    space: SpaceDefinition,
    output_path: Path,
    *,
    dtype=None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    affine = build_nifti_affine_from_space(space)
    if dtype is None:
        array_to_write = array.astype(np.float32, copy=False)
    else:
        array_to_write = array.astype(dtype, copy=False)
    image = nib.Nifti1Image(array_to_write, affine)
    image.header.set_xyzt_units("mm")
    image.header.set_zooms(tuple(float(resolution_um) / _UM_PER_MM for resolution_um in space.resolution_um))
    image.set_qform(affine, code=2)
    image.set_sform(affine, code=2)
    nib.save(image, str(output_path))
    return output_path


def load_nifti_array(input_path: Path) -> np.ndarray:
    image = nib.load(str(input_path))
    return np.asanyarray(image.dataobj)
