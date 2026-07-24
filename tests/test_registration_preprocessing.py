from __future__ import annotations

import unittest

import ants
import numpy as np
from pydantic import ValidationError

from atlasspace.config.preset_models import PreprocessingConfig
from atlasspace.registration.preprocessing import (
    pad_fixed_image,
    preprocess_registration_images,
)


class RegistrationPaddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = ants.from_numpy(
            np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4)),
            spacing=(0.05, 0.05, 0.05),
            origin=(1.0, 2.0, 3.0),
        )

    def test_padding_is_disabled_by_default(self) -> None:
        fixed, moving = preprocess_registration_images(
            self.image,
            self.image,
            PreprocessingConfig(),
        )

        self.assertEqual(fixed.shape, self.image.shape)
        self.assertEqual(moving.shape, self.image.shape)
        self.assertEqual(fixed.origin, self.image.origin)
        self.assertEqual(moving.origin, self.image.origin)

    def test_padding_preserves_original_voxel_physical_coordinates(self) -> None:
        padded = pad_fixed_image(self.image, padding_um=100.0)

        self.assertEqual(padded.shape, (6, 7, 8))
        original_point = ants.transform_index_to_physical_point(
            self.image,
            (0, 0, 0),
        )
        padded_point = ants.transform_index_to_physical_point(
            padded,
            (2, 2, 2),
        )
        np.testing.assert_allclose(padded_point, original_point)
        np.testing.assert_array_equal(
            padded.numpy()[2:4, 2:5, 2:6],
            self.image.numpy(),
        )

    def test_preprocessing_pads_only_the_fixed_image(self) -> None:
        fixed, moving = preprocess_registration_images(
            self.image,
            self.image,
            PreprocessingConfig(fixed_padding_um=100.0),
        )

        self.assertEqual(fixed.shape, (6, 7, 8))
        self.assertEqual(moving.shape, self.image.shape)

    def test_padding_must_be_positive(self) -> None:
        with self.assertRaises(ValidationError):
            PreprocessingConfig(fixed_padding_um=0)


if __name__ == "__main__":
    unittest.main()
