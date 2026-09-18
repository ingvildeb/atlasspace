from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib

from atlasspace.config.image_models import ImageConfig
from atlasspace.registration.result_manifest import (
    REGISTRATION_RESULT_FILENAME,
    RegistrationResultManifest,
    load_registration_result,
    load_registration_result_manifest,
)
from atlasspace.runtime.transforms import TransformSequence
from atlasspace.transforms import transform_segmentation


def _resolve_registration_path(registration_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else registration_dir / path


def _manifest_path_value(path: Path, registration_dir: Path) -> str:
    try:
        return path.relative_to(registration_dir).as_posix()
    except ValueError:
        return str(path)


def _validate_segmentation_name(segmentation_name: str) -> str:
    normalized = segmentation_name.strip()
    if not normalized:
        raise ValueError("segmentation_name must not be empty")
    if Path(normalized).name != normalized or normalized in {".", ".."}:
        raise ValueError(
            "segmentation_name must be a filename-safe name without path separators"
        )
    return normalized


def _validate_source_shape(
    segmentation_path: Path,
    manifest: RegistrationResultManifest,
) -> None:
    expected_shape = manifest.moving_image.space.shape
    if expected_shape is None:
        raise ValueError(
            "The registration manifest does not declare the moving-template shape"
        )
    observed_shape = tuple(int(value) for value in nib.load(str(segmentation_path)).shape)
    if observed_shape != tuple(expected_shape):
        raise ValueError(
            "Segmentation shape does not match the registration moving-template "
            f"shape: {observed_shape} vs {tuple(expected_shape)}"
        )


def _validate_output_shape(
    output_path: Path,
    manifest: RegistrationResultManifest,
) -> None:
    expected_shape = manifest.effective_fixed_space.shape
    if expected_shape is None:
        raise ValueError(
            "The registration manifest does not declare the effective fixed shape"
        )
    observed_shape = tuple(int(value) for value in nib.load(str(output_path)).shape)
    if observed_shape != tuple(expected_shape):
        raise ValueError(
            "Transformed segmentation shape does not match the registration fixed "
            f"shape: {observed_shape} vs {tuple(expected_shape)}"
        )


def _same_source_path(recorded: str, requested: Path) -> bool:
    return os.path.normcase(os.path.abspath(recorded)) == os.path.normcase(
        os.path.abspath(requested)
    )


def _write_manifest_atomic(
    manifest: RegistrationResultManifest,
    registration_dir: Path,
) -> None:
    manifest_path = registration_dir / REGISTRATION_RESULT_FILENAME
    temporary_path = registration_dir / f".{REGISTRATION_RESULT_FILENAME}.partial"
    temporary_path.unlink(missing_ok=True)
    try:
        temporary_path.write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def transform_segmentation_from_registration(
    registration_dir: str | Path,
    segmentation_name: str,
    segmentation_path: str | Path,
    *,
    replace: bool = False,
) -> Path:
    """Transform a template-grid segmentation with a saved registration.

    The segmentation is declared to occupy the registration moving-template
    grid. Its source NIfTI header is not used to infer another orientation.
    Both the source and transformed segmentation mappings in
    ``registration_result.json`` are updated after the output validates.
    """

    resolved_registration_dir = Path(registration_dir)
    resolved_segmentation_path = Path(segmentation_path)
    normalized_name = _validate_segmentation_name(segmentation_name)
    if not resolved_segmentation_path.is_file():
        raise FileNotFoundError(
            f"Segmentation file does not exist: {resolved_segmentation_path}"
        )

    manifest = load_registration_result_manifest(resolved_registration_dir)
    if not manifest.success:
        raise ValueError(
            "Cannot transform a segmentation from an unsuccessful registration"
        )
    _validate_source_shape(resolved_segmentation_path, manifest)

    recorded_source = manifest.moving_image.segmentations.get(normalized_name)
    recorded_output = manifest.transformed_segmentations.get(normalized_name)
    if recorded_source is not None and not _same_source_path(
        recorded_source,
        resolved_segmentation_path,
    ):
        if not replace:
            raise FileExistsError(
                f"Segmentation name {normalized_name!r} is already associated with "
                f"{recorded_source!r}; use replace=True to replace it"
            )

    canonical_output = (
        resolved_registration_dir
        / f"{normalized_name}_WarpedSegmentation.nii.gz"
    )
    output_path = (
        _resolve_registration_path(resolved_registration_dir, recorded_output)
        if recorded_output is not None
        else canonical_output
    )
    if (
        not replace
        and recorded_source is not None
        and recorded_output is not None
        and output_path.is_file()
    ):
        _validate_output_shape(output_path, manifest)
        return output_path

    fixed_reference_value = manifest.fixed_image.normalized_image
    if fixed_reference_value is None:
        fixed_reference_path = Path(manifest.fixed_image.image)
        fixed_reference_space = manifest.fixed_image.space
    else:
        fixed_reference_path = _resolve_registration_path(
            resolved_registration_dir,
            fixed_reference_value,
        )
        fixed_reference_space = manifest.effective_fixed_space
    if not fixed_reference_path.is_file():
        raise FileNotFoundError(
            f"Registration fixed reference does not exist: {fixed_reference_path}"
        )

    result = load_registration_result(resolved_registration_dir)
    transform_sequence = TransformSequence.from_registration_result(result)
    source_config = ImageConfig(
        image_id=normalized_name,
        image=resolved_segmentation_path,
        space=manifest.moving_image.space,
    )
    reference_config = ImageConfig(
        image_id=manifest.fixed_image.image_id,
        image=fixed_reference_path,
        space=fixed_reference_space,
    )
    temporary_output = output_path.with_name(
        f".{output_path.name.removesuffix('.nii.gz')}.partial.nii.gz"
    )
    temporary_output.unlink(missing_ok=True)
    try:
        transform_segmentation(
            source_config,
            transform_sequence,
            reference_config,
            direction="forward",
            interpolation="genericLabel",
            output_path=temporary_output,
            write_intermediates=False,
        )
        _validate_output_shape(temporary_output, manifest)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_output, output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    manifest.moving_image.segmentations[normalized_name] = str(
        resolved_segmentation_path
    )
    manifest.transformed_segmentations[normalized_name] = _manifest_path_value(
        output_path,
        resolved_registration_dir,
    )
    _write_manifest_atomic(manifest, resolved_registration_dir)
    return output_path
