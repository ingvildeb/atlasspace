from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from atlasspace.config.image_models import ImageConfig
from atlasspace.config.preset_models import RegistrationParametersConfig
from atlasspace.config.space_models import SpaceDefinition
from atlasspace.registration.result_manifest import (
    REGISTRATION_RESULT_FILENAME,
    load_registration_result,
    load_registration_result_manifest,
    migrate_legacy_registration_output,
    write_registration_result_manifest,
)
from atlasspace.runtime.registration import RegistrationJob, RegistrationResult
from atlasspace.runtime.transforms import TransformSequence


class RegistrationResultManifestTests(unittest.TestCase):
    def test_write_and_load_registration_result_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            fixed_space = SpaceDefinition(
                space_name="subject_001",
                orientation="las",
                resolution_um=(20.0, 20.0, 20.0),
                shape=(2, 3, 4),
            )
            moving_space = SpaceDefinition(
                space_name="template_p56",
                orientation="lsp",
                resolution_um=(20.0, 20.0, 20.0),
                shape=(5, 6, 7),
            )
            effective_fixed_space = SpaceDefinition(
                space_name="subject_001",
                orientation="lsp",
                axis_labels=("x", "z", "y"),
                resolution_um=(20.0, 20.0, 20.0),
                shape=(2, 4, 3),
            )
            job = RegistrationJob(
                fixed_image_config=ImageConfig(
                    image_id="subject_001",
                    image=Path("/data/subject_001.nii.gz"),
                    space=fixed_space,
                ),
                moving_image_config=ImageConfig(
                    image_id="template_p56",
                    image=Path("/data/template_p56.nii.gz"),
                    space=moving_space,
                ),
                output_dir=output_dir,
                parameters=RegistrationParametersConfig(name="test_preset"),
                orientation_alignment="fixed_to_moving",
            )
            result = RegistrationResult(
                fixed_image_id="subject_001",
                moving_image_id="template_p56",
                preset_name="test_preset",
                output_dir=output_dir,
                success=True,
                declared_fixed_space=fixed_space,
                declared_moving_space=moving_space,
                effective_fixed_space=effective_fixed_space,
                effective_moving_space=moving_space,
                forward_transforms=[
                    output_dir / "ANTsPy_1Warp.nii.gz",
                    output_dir / "ANTsPy_0GenericAffine.mat",
                ],
                inverse_transforms=[
                    output_dir / "ANTsPy_0GenericAffine.mat",
                    output_dir / "ANTsPy_1InverseWarp.nii.gz",
                ],
            )

            manifest_path = write_registration_result_manifest(
                job,
                result,
                fixed_normalized_path=(
                    output_dir / "fixed_normalized_for_registration.nii.gz"
                ),
                moving_normalized_path=(
                    output_dir / "moving_normalized_for_registration.nii.gz"
                ),
            )

            self.assertEqual(manifest_path, output_dir / REGISTRATION_RESULT_FILENAME)
            manifest = load_registration_result_manifest(output_dir)
            self.assertEqual(manifest.fixed_image.space, fixed_space)
            self.assertEqual(manifest.effective_fixed_space, effective_fixed_space)
            self.assertEqual(manifest.orientation_alignment, "fixed_to_moving")
            self.assertEqual(
                manifest.forward_transforms,
                ["ANTsPy_1Warp.nii.gz", "ANTsPy_0GenericAffine.mat"],
            )

    def test_migrate_legacy_registration_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            self._write_nifti(
                output_dir / "fixed_normalized_for_registration.nii.gz",
                shape=(2, 4, 3),
            )
            self._write_nifti(
                output_dir / "moving_normalized_for_registration.nii.gz",
                shape=(5, 6, 7),
            )
            for filename in (
                "ANTsPy_0GenericAffine.mat",
                "ANTsPy_1Warp.nii.gz",
                "ANTsPy_1InverseWarp.nii.gz",
                "ANTsPy_Warped.nii.gz",
                "ANTsPy_InverseWarped.nii.gz",
                "labels_WarpedSegmentation.nii.gz",
            ):
                (output_dir / filename).touch()

            (output_dir / "registration_summary.txt").write_text(
                "\n".join(
                    (
                        "success=True",
                        "fixed_image_id=subject_001",
                        "moving_image_id=template_p56",
                        "fixed_image=/data/subject_001.nii.gz",
                        "moving_image=/data/template_p56.nii.gz",
                        "fixed_space_name=subject_001",
                        "moving_space_name=template_p56",
                        "configured_fixed_resolution_um=(20.0, 20.0, 20.0)",
                        "configured_moving_resolution_um=(20.0, 20.0, 20.0)",
                        "effective_fixed_resolution_um=(20.0, 20.0, 20.0)",
                        "effective_moving_resolution_um=(20.0, 20.0, 20.0)",
                        "configured_fixed_orientation=las",
                        "configured_moving_orientation=lsp",
                        "effective_fixed_orientation=lsp",
                        "effective_moving_orientation=lsp",
                        "preset_name=tuned_syn_cc",
                        "runtime_seconds=123.5",
                        "error_message=None",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            manifest_path = migrate_legacy_registration_output(
                output_dir,
                fixed_image="/new/batches/batch001/subjects/subject_001/ch1_native20um.nii.gz",
            )

            self.assertEqual(manifest_path, output_dir / REGISTRATION_RESULT_FILENAME)
            manifest = load_registration_result_manifest(output_dir)
            self.assertEqual(manifest.orientation_alignment, "fixed_to_moving")
            self.assertEqual(manifest.fixed_image.space.orientation, "las")
            self.assertEqual(manifest.fixed_image.space.shape, (2, 3, 4))
            self.assertEqual(manifest.effective_fixed_space.orientation, "lsp")
            self.assertEqual(manifest.effective_fixed_space.axis_labels, ("x", "z", "y"))
            self.assertEqual(manifest.effective_fixed_space.shape, (2, 4, 3))
            self.assertEqual(
                manifest.forward_transforms,
                ["ANTsPy_1Warp.nii.gz", "ANTsPy_0GenericAffine.mat"],
            )
            self.assertEqual(
                manifest.inverse_transforms,
                ["ANTsPy_0GenericAffine.mat", "ANTsPy_1InverseWarp.nii.gz"],
            )
            self.assertEqual(
                manifest.transformed_segmentations,
                {"labels": "labels_WarpedSegmentation.nii.gz"},
            )
            self.assertIsNotNone(manifest.migration)
            self.assertEqual(
                manifest.fixed_image.image,
                "/new/batches/batch001/subjects/subject_001/ch1_native20um.nii.gz",
            )
            self.assertEqual(
                manifest.migration.original_fixed_image,
                "/data/subject_001.nii.gz",
            )
            self.assertIsNone(manifest.migration.original_moving_image)

            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(raw_manifest["schema_version"], 1)
            self.assertNotIn(str(output_dir), raw_manifest["forward_transforms"])

            result = load_registration_result(output_dir)
            self.assertEqual(result.declared_fixed_space.shape, (2, 3, 4))
            self.assertEqual(
                result.forward_transforms,
                [
                    output_dir / "ANTsPy_1Warp.nii.gz",
                    output_dir / "ANTsPy_0GenericAffine.mat",
                ],
            )

            sequence = TransformSequence.from_registration_output(output_dir)
            self.assertEqual(sequence.source_space.space_name, "template_p56")
            self.assertEqual(sequence.target_space.space_name, "subject_001")
            self.assertEqual(sequence.target_transform_space.orientation, "lsp")

    def test_migration_does_not_overwrite_manifest_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            (output_dir / REGISTRATION_RESULT_FILENAME).write_text("{}", encoding="utf-8")
            (output_dir / "registration_summary.txt").write_text(
                "success=True\n",
                encoding="utf-8",
            )

            with self.assertRaises(FileExistsError):
                migrate_legacy_registration_output(output_dir)

    @staticmethod
    def _write_nifti(path: Path, shape: tuple[int, int, int]) -> None:
        image = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine=np.eye(4))
        nib.save(image, str(path))


if __name__ == "__main__":
    unittest.main()
