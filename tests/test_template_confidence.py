import unittest

import numpy as np

from atlasspace.template.confidence import _compute_residual_array


class TemplateConfidenceTests(unittest.TestCase):
    def test_subject_deficit_relative_ignores_excess_subject_signal(self) -> None:
        subject = np.array([0.4, 1.0, 0.8], dtype=np.float32)
        template = np.array([0.8, 0.8, 0.8], dtype=np.float32)

        residual = _compute_residual_array(
            subject,
            template,
            residual_mode="subject_deficit_relative",
            template_relative_floor=0.2,
        )

        np.testing.assert_allclose(
            residual,
            np.array([0.4, 0.0, 0.0], dtype=np.float32),
        )

    def test_relative_mode_remains_two_sided(self) -> None:
        subject = np.array([0.4, 1.0], dtype=np.float32)
        template = np.array([0.8, 0.8], dtype=np.float32)

        residual = _compute_residual_array(
            subject,
            template,
            residual_mode="relative",
            template_relative_floor=0.2,
        )

        np.testing.assert_allclose(
            residual,
            np.array([0.4, 0.2], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
