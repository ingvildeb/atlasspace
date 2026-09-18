from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

from atlasspace.config.image_models import ImageConfig
from atlasspace.config.space_models import SpaceDefinition
from atlasspace.registration import (
    load_registration_result_manifest,
    transform_segmentation_from_registration,
)
from atlasspace.registration.result_manifest import (
    REGISTRATION_RESULT_FILENAME,
    RegistrationManifestImage,
    RegistrationResultManifest,
)


def _write_nifti(path: Path, shape: tuple[int, int, int], value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full(shape, value, dtype=np.int16)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _write_registration(registration_dir: Path) -> tuple[SpaceDefinition, SpaceDefinition]:
    moving_space = SpaceDefinition(
        space_name="template_p56",
        orientation="lsp",
        resolution_um=(20.0, 20.0, 20.0),
        shape=(3, 4, 5),
    )
    fixed_space = SpaceDefinition(
        space_name="subject_001",
        orientation="lsp",
        resolution_um=(20.0, 20.0, 20.0),
        shape=(2, 3, 4),
    )
    registration_dir.mkdir(parents=True)
    _write_nifti(
        registration_dir / "fixed_normalized_for_registration.nii.gz",
        fixed_space.shape,
    )
    _write_nifti(
        registration_dir / "moving_normalized_for_registration.nii.gz",
        moving_space.shape,
    )
    for filename in (
        "ANTsPy_0GenericAffine.mat",
        "ANTsPy_1Warp.nii.gz",
        "ANTsPy_1InverseWarp.nii.gz",
    ):
        (registration_dir / filename).touch()

    manifest = RegistrationResultManifest(
        success=True,
        preset_name="tuned_syn_cc",
        orientation_alignment="none",
        fixed_image=RegistrationManifestImage(
            image_id="subject_001",
            image="/data/subject_001.nii.gz",
            normalized_image="fixed_normalized_for_registration.nii.gz",
            space=fixed_space,
        ),
        moving_image=RegistrationManifestImage(
            image_id="template_p56",
            image="/data/template_p56.nii.gz",
            normalized_image="moving_normalized_for_registration.nii.gz",
            space=moving_space,
        ),
        effective_fixed_space=fixed_space,
        effective_moving_space=moving_space,
        forward_transforms=[
            "ANTsPy_1Warp.nii.gz",
            "ANTsPy_0GenericAffine.mat",
        ],
        inverse_transforms=[
            "ANTsPy_0GenericAffine.mat",
            "ANTsPy_1InverseWarp.nii.gz",
        ],
    )
    (registration_dir / REGISTRATION_RESULT_FILENAME).write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return moving_space, fixed_space


def _fake_transform(fixed_shape: tuple[int, int, int]) -> Mock:
    def transform(
        image_config: ImageConfig,
        transform_sequence,
        reference_config: ImageConfig,
        *,
        direction: str,
        interpolation: str,
        output_path: Path,
        write_intermediates: bool,
    ) -> ImageConfig:
        assert image_config.space.orientation == "lsp"
        assert image_config.space.shape == (3, 4, 5)
        assert reference_config.space.shape == fixed_shape
        assert direction == "forward"
        assert interpolation == "genericLabel"
        assert write_intermediates is False
        _write_nifti(output_path, fixed_shape, value=7)
        return ImageConfig(
            image_id=image_config.image_id,
            image=output_path,
            space=reference_config.space,
        )

    return Mock(side_effect=transform)


def test_add_transforms_and_updates_both_manifest_mappings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_dir = tmp_path / "registration"
    _, fixed_space = _write_registration(registration_dir)
    source = tmp_path / "hemispheres_faulty_header.nii.gz"
    _write_nifti(source, (3, 4, 5))
    transform = _fake_transform(fixed_space.shape)
    monkeypatch.setattr(
        "atlasspace.registration.segmentation_transformation.transform_segmentation",
        transform,
    )

    output = transform_segmentation_from_registration(
        registration_dir,
        "hemispheres",
        source,
    )

    assert output == registration_dir / "hemispheres_WarpedSegmentation.nii.gz"
    assert output.is_file()
    manifest = load_registration_result_manifest(registration_dir)
    assert manifest.moving_image.segmentations["hemispheres"] == str(source)
    assert (
        manifest.transformed_segmentations["hemispheres"]
        == "hemispheres_WarpedSegmentation.nii.gz"
    )
    transform.assert_called_once()


def test_add_is_a_noop_when_same_source_and_valid_output_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_dir = tmp_path / "registration"
    _, fixed_space = _write_registration(registration_dir)
    source = tmp_path / "hemispheres.nii.gz"
    output = registration_dir / "hemispheres_WarpedSegmentation.nii.gz"
    _write_nifti(source, (3, 4, 5))
    _write_nifti(output, fixed_space.shape)
    manifest = load_registration_result_manifest(registration_dir)
    manifest.moving_image.segmentations["hemispheres"] = str(source)
    manifest.transformed_segmentations["hemispheres"] = output.name
    (registration_dir / REGISTRATION_RESULT_FILENAME).write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    transform = Mock(side_effect=AssertionError("transform should not run"))
    monkeypatch.setattr(
        "atlasspace.registration.segmentation_transformation.transform_segmentation",
        transform,
    )

    observed = transform_segmentation_from_registration(
        registration_dir,
        "hemispheres",
        source,
    )

    assert observed == output
    transform.assert_not_called()


@pytest.mark.parametrize("manifest_has_output", [False, True])
def test_add_repairs_interrupted_output_and_manifest_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_has_output: bool,
) -> None:
    registration_dir = tmp_path / "registration"
    _, fixed_space = _write_registration(registration_dir)
    source = tmp_path / "hemispheres.nii.gz"
    output = registration_dir / "hemispheres_WarpedSegmentation.nii.gz"
    _write_nifti(source, (3, 4, 5))
    manifest = load_registration_result_manifest(registration_dir)
    manifest.moving_image.segmentations["hemispheres"] = str(source)
    if manifest_has_output:
        manifest.transformed_segmentations["hemispheres"] = output.name
    else:
        _write_nifti(output, fixed_space.shape, value=2)
    (registration_dir / REGISTRATION_RESULT_FILENAME).write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    transform = _fake_transform(fixed_space.shape)
    monkeypatch.setattr(
        "atlasspace.registration.segmentation_transformation.transform_segmentation",
        transform,
    )

    observed = transform_segmentation_from_registration(
        registration_dir,
        "hemispheres",
        source,
    )

    assert observed == output
    assert np.all(np.asanyarray(nib.load(str(output)).dataobj) == 7)
    repaired = load_registration_result_manifest(registration_dir)
    assert repaired.transformed_segmentations["hemispheres"] == output.name
    transform.assert_called_once()


def test_add_rejects_an_existing_name_from_another_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_dir = tmp_path / "registration"
    _, fixed_space = _write_registration(registration_dir)
    original_source = tmp_path / "hemispheres_v1.nii.gz"
    replacement_source = tmp_path / "hemispheres_v2.nii.gz"
    output = registration_dir / "hemispheres_WarpedSegmentation.nii.gz"
    _write_nifti(original_source, (3, 4, 5))
    _write_nifti(replacement_source, (3, 4, 5))
    _write_nifti(output, fixed_space.shape)
    manifest = load_registration_result_manifest(registration_dir)
    manifest.moving_image.segmentations["hemispheres"] = str(original_source)
    manifest.transformed_segmentations["hemispheres"] = output.name
    (registration_dir / REGISTRATION_RESULT_FILENAME).write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    transform = _fake_transform(fixed_space.shape)
    monkeypatch.setattr(
        "atlasspace.registration.segmentation_transformation.transform_segmentation",
        transform,
    )

    with pytest.raises(FileExistsError, match="replace=True"):
        transform_segmentation_from_registration(
            registration_dir,
            "hemispheres",
            replacement_source,
        )

    transform.assert_not_called()


def test_replace_replaces_output_and_manifest_without_a_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_dir = tmp_path / "registration"
    _, fixed_space = _write_registration(registration_dir)
    original_source = tmp_path / "labels_v1.nii.gz"
    replacement_source = tmp_path / "labels_v2.nii.gz"
    output = registration_dir / "labels_WarpedSegmentation.nii.gz"
    _write_nifti(original_source, (3, 4, 5))
    _write_nifti(replacement_source, (3, 4, 5))
    _write_nifti(output, fixed_space.shape, value=1)
    manifest = load_registration_result_manifest(registration_dir)
    manifest.moving_image.segmentations["labels"] = str(original_source)
    manifest.transformed_segmentations["labels"] = output.name
    (registration_dir / REGISTRATION_RESULT_FILENAME).write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "atlasspace.registration.segmentation_transformation.transform_segmentation",
        _fake_transform(fixed_space.shape),
    )

    observed = transform_segmentation_from_registration(
        registration_dir,
        "labels",
        replacement_source,
        replace=True,
    )

    assert observed == output
    assert np.all(np.asanyarray(nib.load(str(output)).dataobj) == 7)
    replaced = load_registration_result_manifest(registration_dir)
    assert replaced.moving_image.segmentations["labels"] == str(replacement_source)
    assert replaced.transformed_segmentations["labels"] == output.name
    assert not (registration_dir / "segmentation_history").exists()
    assert not list(registration_dir.glob("*previous*"))


def test_wrong_source_shape_fails_without_changing_manifest(
    tmp_path: Path,
) -> None:
    registration_dir = tmp_path / "registration"
    _write_registration(registration_dir)
    source = tmp_path / "wrong_grid.nii.gz"
    _write_nifti(source, (2, 2, 2))
    manifest_path = registration_dir / REGISTRATION_RESULT_FILENAME
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="shape"):
        transform_segmentation_from_registration(
            registration_dir,
            "hemispheres",
            source,
        )

    assert manifest_path.read_bytes() == original_manifest
    assert not (registration_dir / "hemispheres_WarpedSegmentation.nii.gz").exists()


def test_failed_replace_preserves_existing_output_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_dir = tmp_path / "registration"
    _, fixed_space = _write_registration(registration_dir)
    original_source = tmp_path / "labels_v1.nii.gz"
    replacement_source = tmp_path / "labels_v2.nii.gz"
    output = registration_dir / "labels_WarpedSegmentation.nii.gz"
    _write_nifti(original_source, (3, 4, 5))
    _write_nifti(replacement_source, (3, 4, 5))
    _write_nifti(output, fixed_space.shape, value=3)
    manifest = load_registration_result_manifest(registration_dir)
    manifest.moving_image.segmentations["labels"] = str(original_source)
    manifest.transformed_segmentations["labels"] = output.name
    manifest_path = registration_dir / REGISTRATION_RESULT_FILENAME
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_bytes()
    original_output = output.read_bytes()

    def fail_transform(*args, **kwargs):
        raise RuntimeError("simulated transform failure")

    monkeypatch.setattr(
        "atlasspace.registration.segmentation_transformation.transform_segmentation",
        fail_transform,
    )

    with pytest.raises(RuntimeError, match="simulated transform failure"):
        transform_segmentation_from_registration(
            registration_dir,
            "labels",
            replacement_source,
            replace=True,
        )

    assert manifest_path.read_bytes() == original_manifest
    assert output.read_bytes() == original_output
    assert not list(registration_dir.glob("*.partial.nii.gz"))
