from __future__ import annotations

import unittest

import numpy as np

from atlasspace.config.space_models import SpaceDefinition
from atlasspace.image.symmetry import (
    build_hemisphere_map,
    infer_left_right_axis,
)


def _space(*, orientation: str, shape: tuple[int, int, int] | None) -> SpaceDefinition:
    return SpaceDefinition(
        space_name="test_space",
        orientation=orientation,
        resolution_um=(20.0, 20.0, 20.0),
        shape=shape,
    )


class HemisphereMapTests(unittest.TestCase):
    def test_infer_left_right_axis_normalizes_orientation(self) -> None:
        self.assertEqual(infer_left_right_axis(" LSP "), (0, True))
        self.assertEqual(infer_left_right_axis("ASR"), (2, False))

    def test_lsp_odd_assigns_central_plane_left(self) -> None:
        result = build_hemisphere_map(
            _space(orientation="lsp", shape=(5, 2, 3))
        )

        self.assertEqual(result.dtype, np.dtype(np.uint8))
        self.assertEqual(set(np.unique(result)), {1, 2})
        self.assertTrue(np.all(result[:3] == 1))
        self.assertTrue(np.all(result[3:] == 2))

    def test_lsp_even_splits_equally(self) -> None:
        result = build_hemisphere_map(
            _space(orientation="lsp", shape=(4, 2, 3))
        )

        self.assertTrue(np.all(result[:2] == 1))
        self.assertTrue(np.all(result[2:] == 2))

    def test_asr_uses_third_axis_and_low_indices_right(self) -> None:
        result = build_hemisphere_map(
            _space(orientation="asr", shape=(2, 3, 5))
        )

        self.assertTrue(np.all(result[:, :, :2] == 2))
        self.assertTrue(np.all(result[:, :, 2:] == 1))

    def test_requires_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "space.shape is required"):
            build_hemisphere_map(_space(orientation="lsp", shape=None))


if __name__ == "__main__":
    unittest.main()
