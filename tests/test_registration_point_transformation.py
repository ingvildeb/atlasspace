from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from atlasspace.io.nifti import save_array_as_nifti_in_declared_space
from atlasspace.transforms import transform_points_from_registration
from atlasspace.transforms.registration_points import (
    _continuous_indices_to_physical,
    _physical_to_continuous_indices,
)


class RegistrationPointTransformationTests(unittest.TestCase):
    def _registration(self, root: Path) -> Path:
        shape = (8, 9, 10)
        for role in ("fixed", "moving"):
            save_array_as_nifti_in_declared_space(
                np.zeros(shape, dtype=np.uint8),
                root / f"{role}.nii.gz",
                "las",
                (20.0, 20.0, 20.0),
            )
        for name in ("forward.mat", "inverse.mat"):
            (root / name).touch()
        space = {
            "space_name": "test",
            "orientation": "las",
            "axis_labels": ["x", "y", "z"],
            "units": "voxel",
            "resolution_um": [20.0, 20.0, 20.0],
            "shape": list(shape),
        }
        manifest = {
            "schema_version": 1,
            "success": True,
            "preset_name": "test",
            "parameters_snapshot": None,
            "orientation_alignment": "none",
            "fixed_image": {
                "image_id": "fixed",
                "image": "native_fixed.nii.gz",
                "space": space,
                "normalized_image": "fixed.nii.gz",
                "segmentations": {},
            },
            "moving_image": {
                "image_id": "moving",
                "image": "native_moving.nii.gz",
                "space": space,
                "normalized_image": "moving.nii.gz",
                "segmentations": {},
            },
            "effective_fixed_space": space,
            "effective_moving_space": space,
            "forward_transforms": ["forward.mat"],
            "inverse_transforms": ["inverse.mat"],
            "transformed_segmentations": {},
        }
        (root / "registration_result.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return root

    def test_geometry_helpers_round_trip_fractional_indices_with_direction(self) -> None:
        indices = np.array([[0.25, 2.5, 7.75], [4.0, 1.0, 0.0]])
        origin = np.array([10.0, -3.0, 5.0])
        spacing = np.array([0.02, 0.03, 0.04])
        direction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        physical = _continuous_indices_to_physical(
            indices, origin=origin, spacing=spacing, direction=direction
        )
        recovered = _physical_to_continuous_indices(
            physical, origin=origin, spacing=spacing, direction=direction
        )
        np.testing.assert_allclose(recovered, indices)

    def test_roles_translation_and_chunking_select_transform_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            registration_dir = self._registration(Path(tmp_dir))
            points = np.array([[0.25, 1.5, 2.75], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            directions: list[str] = []

            def translated(physical, sequence, *, direction):
                directions.append(direction)
                delta = np.array([0.02, 0.0, 0.0])
                return physical + delta if direction == "forward" else physical - delta

            with patch(
                "atlasspace.transforms.registration_points.transform_points",
                side_effect=translated,
            ):
                forward = transform_points_from_registration(
                    points,
                    registration_dir,
                    source_role="fixed",
                    target_role="moving",
                    chunk_size=2,
                )
                inverse = transform_points_from_registration(
                    forward,
                    registration_dir,
                    source_role="moving",
                    target_role="fixed",
                    chunk_size=None,
                )
            np.testing.assert_allclose(np.abs(forward[:, 0] - points[:, 0]), 1.0)
            np.testing.assert_allclose(forward[:, 1:], points[:, 1:])
            np.testing.assert_allclose(inverse, points)
            self.assertEqual(directions, ["forward", "forward", "inverse"])

    def test_validation_rejects_same_roles_and_nonfinite_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            registration_dir = self._registration(Path(tmp_dir))
            with self.assertRaisesRegex(ValueError, "distinct"):
                transform_points_from_registration(
                    np.zeros((1, 3)),
                    registration_dir,
                    source_role="fixed",
                    target_role="fixed",
                )
            with self.assertRaisesRegex(ValueError, "finite"):
                transform_points_from_registration(
                    np.array([[np.nan, 0.0, 0.0]]),
                    registration_dir,
                    source_role="fixed",
                    target_role="moving",
                )

    def test_missing_transform_and_manifest_header_disagreement_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            registration_dir = self._registration(Path(tmp_dir))
            (registration_dir / "forward.mat").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "transform"):
                transform_points_from_registration(
                    np.zeros((1, 3)),
                    registration_dir,
                    source_role="fixed",
                    target_role="moving",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            registration_dir = self._registration(Path(tmp_dir))
            manifest_path = registration_dir / "registration_result.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["effective_fixed_space"]["shape"] = [7, 9, 10]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shape"):
                transform_points_from_registration(
                    np.zeros((1, 3)),
                    registration_dir,
                    source_role="fixed",
                    target_role="moving",
                )


if __name__ == "__main__":
    unittest.main()
