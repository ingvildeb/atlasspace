from __future__ import annotations

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np

from atlasspace.image.space_validation import (
    assert_nifti_header_matches_declared_space,
)
from atlasspace.registration.result_manifest import (
    RegistrationResultManifest,
    load_registration_result_manifest,
)
from atlasspace.runtime.transforms import TransformSequence
from atlasspace.transforms.application import transform_points


RegistrationRole = Literal["fixed", "moving"]


def _validate_points_array(
    points_xyz: np.ndarray,
    *,
    label: str,
    require_finite: bool = True,
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} must have shape (N, 3); got {points.shape}.")
    if require_finite and not np.all(np.isfinite(points)):
        raise ValueError(f"{label} must contain only finite values.")
    return points


def _continuous_indices_to_physical(
    indices_xyz: np.ndarray,
    *,
    origin: np.ndarray,
    spacing: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    indices = _validate_points_array(indices_xyz, label="indices_xyz")
    return origin + (indices * spacing) @ direction.T


def _physical_to_continuous_indices(
    points_xyz: np.ndarray,
    *,
    origin: np.ndarray,
    spacing: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    points = _validate_points_array(
        points_xyz,
        label="points_xyz",
        require_finite=False,
    )
    return ((points - origin) @ np.linalg.inv(direction).T) / spacing


def _resolved_manifest_path(registration_dir: Path, value: str | None, *, label: str) -> Path:
    if value is None:
        raise ValueError(f"Registration manifest does not define {label}.")
    path = Path(value)
    path = path if path.is_absolute() else registration_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} not found: {path}")
    return path


def _normalized_image_path(
    registration_dir: Path,
    manifest: RegistrationResultManifest,
    role: RegistrationRole,
) -> Path:
    image = manifest.fixed_image if role == "fixed" else manifest.moving_image
    return _resolved_manifest_path(
        registration_dir,
        image.normalized_image,
        label=f"normalized {role} image",
    )


def _validate_normalized_image_geometry(
    path: Path,
    manifest: RegistrationResultManifest,
    role: RegistrationRole,
) -> None:
    space = (
        manifest.effective_fixed_space
        if role == "fixed"
        else manifest.effective_moving_space
    )
    assert_nifti_header_matches_declared_space(
        path,
        orientation=space.orientation,
        resolution_um=space.resolution_um,
    )
    observed_shape = tuple(int(value) for value in nib.load(str(path)).shape)
    if len(observed_shape) != 3:
        raise ValueError(f"Normalized {role} image must be 3D: {path}")
    if space.shape is not None and observed_shape != tuple(space.shape):
        raise ValueError(
            f"Normalized {role} image shape does not match the registration manifest: "
            f"header={observed_shape}, manifest={space.shape}."
        )


def _nifti_lps_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    affine_ras = np.asarray(nib.load(str(path)).affine, dtype=np.float64)
    if affine_ras.shape != (4, 4) or not np.all(np.isfinite(affine_ras)):
        raise ValueError(f"Normalized registration image has an invalid affine: {path}")
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_lps = ras_to_lps @ affine_ras
    basis = affine_lps[:3, :3]
    origin = affine_lps[:3, 3]
    spacing = np.linalg.norm(basis, axis=0)
    if np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError("Normalized registration image has invalid spacing.")
    direction = basis / spacing[np.newaxis, :]
    if origin.shape != (3,) or spacing.shape != (3,) or direction.shape != (3, 3):
        raise ValueError("Normalized registration image has invalid 3D geometry.")
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(spacing)):
        raise ValueError("Normalized registration image has non-finite geometry.")
    if np.any(spacing <= 0) or not np.all(np.isfinite(direction)):
        raise ValueError("Normalized registration image has invalid spacing or direction.")
    if np.isclose(np.linalg.det(direction), 0.0):
        raise ValueError("Normalized registration image direction matrix is singular.")
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6):
        raise ValueError("Normalized registration image direction matrix is not orthonormal.")
    return origin, spacing, direction


def transform_points_from_registration(
    source_indices_xyz: np.ndarray,
    registration_dir: Path | str,
    *,
    source_role: RegistrationRole,
    target_role: RegistrationRole,
    chunk_size: int | None = 250_000,
) -> np.ndarray:
    """Map continuous voxel indices between normalized registration images.

    The source and returned coordinates are zero-based continuous voxel indices.
    Image headers provide the conversion to and from the LPS+ physical millimetres
    required by the ANTs point-transformation interface.
    """
    source_indices = _validate_points_array(
        source_indices_xyz,
        label="source_indices_xyz",
    )
    valid_roles = {"fixed", "moving"}
    if source_role not in valid_roles or target_role not in valid_roles:
        raise ValueError("source_role and target_role must be 'fixed' or 'moving'.")
    if source_role == target_role:
        raise ValueError("source_role and target_role must be distinct.")
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer or None.")

    resolved_dir = Path(registration_dir)
    manifest = load_registration_result_manifest(resolved_dir)
    if not manifest.success:
        detail = f": {manifest.error_message}" if manifest.error_message else "."
        raise ValueError(f"Registration was not successful{detail}")

    source_path = _normalized_image_path(resolved_dir, manifest, source_role)
    target_path = _normalized_image_path(resolved_dir, manifest, target_role)
    _validate_normalized_image_geometry(source_path, manifest, source_role)
    _validate_normalized_image_geometry(target_path, manifest, target_role)

    sequence = TransformSequence.from_registration_output(resolved_dir)
    direction = "forward" if source_role == "fixed" else "inverse"
    required_paths = sequence.paths_for_direction(direction)
    missing_paths = [path for path in required_paths if not Path(path).is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required registration transform(s) not found: {missing}")

    source_geometry = _nifti_lps_geometry(source_path)
    target_geometry = _nifti_lps_geometry(target_path)
    if source_indices.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    step = source_indices.shape[0] if chunk_size is None else chunk_size
    transformed_indices = np.empty(source_indices.shape, dtype=np.float64)
    for start in range(0, source_indices.shape[0], step):
        stop = min(start + step, source_indices.shape[0])
        source_chunk = source_indices[start:stop]
        physical_chunk = _continuous_indices_to_physical(
            source_chunk,
            origin=source_geometry[0],
            spacing=source_geometry[1],
            direction=source_geometry[2],
        )
        transformed_physical = transform_points(
            physical_chunk,
            sequence,
            direction=direction,
        )
        transformed_indices[start:stop] = _physical_to_continuous_indices(
            transformed_physical,
            origin=target_geometry[0],
            spacing=target_geometry[1],
            direction=target_geometry[2],
        )
    return transformed_indices
