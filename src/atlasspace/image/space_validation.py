from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError(
        "atlasspace.image.space_validation requires nibabel. "
        "Install atlasspace with `pip install -e .` or otherwise ensure nibabel is available."
    ) from exc

from atlasspace.config.space_models import SpaceDefinition
from atlasspace.io.nifti import (
    _BRAINGLOBE_ORIGIN_TO_RAS,
    _UM_PER_MM,
    build_nifti_affine_from_declared_space,
)


def is_isotropic(space: SpaceDefinition) -> bool:
    x, y, z = space.resolution_um
    return x == y == z


def validate_isotropic_space(space: SpaceDefinition) -> None:
    if not is_isotropic(space):
        raise ValueError(
            "Registration currently supports only isotropic image spaces. "
            f"Got resolution_um={space.resolution_um} for space "
            f"{space.space_name or '<unnamed>'}."
        )


_RAS_TO_BRAINGLOBE_ORIGIN = {
    (world_axis, int(sign)): orientation_letter
    for orientation_letter, (world_axis, sign) in _BRAINGLOBE_ORIGIN_TO_RAS.items()
}


def _orientation_from_affine(affine: np.ndarray) -> str | None:
    orientation_letters: list[str] = []
    seen_world_axes: set[int] = set()

    for voxel_axis in range(3):
        axis_vector = np.asarray(affine[:3, voxel_axis], dtype=np.float64)
        if np.allclose(axis_vector, 0.0):
            return None

        world_axis = int(np.argmax(np.abs(axis_vector)))
        if world_axis in seen_world_axes:
            return None
        seen_world_axes.add(world_axis)

        dominant_value = float(axis_vector[world_axis])
        if np.isclose(dominant_value, 0.0):
            return None

        sign = 1 if dominant_value > 0 else -1
        orientation_letter = _RAS_TO_BRAINGLOBE_ORIGIN.get((world_axis, sign))
        if orientation_letter is None:
            return None
        orientation_letters.append(orientation_letter)

    return "".join(orientation_letters)


def check_nifti_header_matches_declared_space(
    nifti_path: Path,
    orientation: str,
    resolution_um: float | int | tuple[float, float, float] | list[float],
) -> dict[str, object]:
    image = nib.load(str(nifti_path))
    header = image.header
    expected_affine = build_nifti_affine_from_declared_space(
        shape=image.shape[:3],
        orientation=orientation,
        resolution_um=resolution_um,
    )

    observed_affine = np.asarray(image.affine, dtype=np.float64)
    observed_orientation = _orientation_from_affine(observed_affine)
    observed_zooms = tuple(float(value) for value in header.get_zooms()[:3])
    expected_resolution_mm = float(expected_affine[np.nonzero(expected_affine[:3, :3])][0])
    expected_resolution_mm = abs(expected_resolution_mm)
    expected_zooms = (expected_resolution_mm, expected_resolution_mm, expected_resolution_mm)
    spatial_units, temporal_units = header.get_xyzt_units()

    qform_affine, qform_code = header.get_qform(coded=True)
    sform_affine, sform_code = header.get_sform(coded=True)

    affine_matches = np.allclose(observed_affine, expected_affine)
    orientation_matches = observed_orientation == orientation.strip().lower()
    zooms_match = np.allclose(observed_zooms, expected_zooms)
    units_match = spatial_units == "mm"
    qform_matches = qform_affine is not None and np.allclose(qform_affine, expected_affine)
    sform_matches = sform_affine is not None and np.allclose(sform_affine, expected_affine)

    mismatches: list[str] = []
    if not affine_matches:
        mismatches.append("affine")
    if not orientation_matches:
        mismatches.append("orientation")
    if not zooms_match:
        mismatches.append("resolution")
    if not units_match:
        mismatches.append("units")
    if not qform_matches:
        mismatches.append("qform")
    if not sform_matches:
        mismatches.append("sform")

    warnings: list[str] = []
    if qform_code == 0:
        warnings.append("qform_code_zero")
    if sform_code == 0:
        warnings.append("sform_code_zero")

    return {
        "matches": not mismatches,
        "path": Path(nifti_path),
        "expected_orientation": orientation.strip().lower(),
        "observed_orientation": observed_orientation,
        "expected_resolution_um": resolution_um,
        "expected_zooms_mm": expected_zooms,
        "observed_zooms_mm": observed_zooms,
        "expected_units": "mm",
        "observed_units": spatial_units,
        "observed_temporal_units": temporal_units,
        "expected_affine": expected_affine,
        "observed_affine": observed_affine,
        "affine_matches": affine_matches,
        "orientation_matches": orientation_matches,
        "resolution_matches": zooms_match,
        "units_match": units_match,
        "qform_matches": qform_matches,
        "sform_matches": sform_matches,
        "observed_qform_code": qform_code,
        "observed_sform_code": sform_code,
        "mismatches": mismatches,
        "warnings": warnings,
    }


def assert_nifti_header_matches_declared_space(
    nifti_path: Path,
    orientation: str,
    resolution_um: float | int | tuple[float, float, float] | list[float],
) -> None:
    result = check_nifti_header_matches_declared_space(
        nifti_path=nifti_path,
        orientation=orientation,
        resolution_um=resolution_um,
    )
    if result["matches"]:
        return

    mismatch_summary = ", ".join(result["mismatches"])
    raise ValueError(
        f"NIfTI header does not match declared space for {nifti_path}: {mismatch_summary}. "
        f"Observed orientation={result['observed_orientation']!r}, "
        f"observed_zooms_mm={result['observed_zooms_mm']!r}, "
        f"observed_units={result['observed_units']!r}, "
        f"qform_code={result['observed_qform_code']!r}, "
        f"sform_code={result['observed_sform_code']!r}."
    )
