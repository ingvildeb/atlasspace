from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError(
        "atlasbuilder.io.nifti requires nibabel. "
        "Install atlasbuilder with nibabel available to write normalized NIfTI files."
    ) from exc

from atlasbuilder.config.space_models import SpaceDefinition


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
        affine[world_axis, axis_index] = sign * float(resolution_um)

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
    image.set_qform(affine)
    image.set_sform(affine)
    nib.save(image, str(output_path))
    return output_path


def load_nifti_array(input_path: Path) -> np.ndarray:
    image = nib.load(str(input_path))
    return np.asanyarray(image.dataobj)
