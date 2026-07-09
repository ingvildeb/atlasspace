from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atlasspace.config.config_loading import load_registration_plan
from atlasspace.registration.job_building import build_jobs_from_plan


class RegistrationJobSpecTomlTests(unittest.TestCase):
    def test_single_plan_resolves_defaults_and_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "single.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    registration_presets = ["baseline_syn_kimlab"]
                    orientation_alignment = "fixed_to_moving"
                    write_input_images = true
                    output_dir = "outputs/single_run"

                    [image_defaults]
                    resolution_um = 20.0

                    [moving_segmentations]
                    enabled = true
                    interpolation = "genericLabel"
                    output_subdir = "moving_segmentations"
                    write_intermediates = false

                    [images.fixed]
                    image = "fixed.nii.gz"
                    orientation = "las"

                    [images.moving]
                    image = "moving.nii.gz"
                    orientation = "lsp"

                    [images.moving.segmentations]
                    brain_mask = "moving_mask.nii.gz"

                    [single]
                    fixed_image = "fixed"
                    moving_image = "moving"
                    """
                ).strip(),
                encoding="utf-8",
            )

            plan = load_registration_plan(config_path)

            self.assertEqual(plan.mode, "single")
            self.assertEqual(plan.orientation_alignment, "fixed_to_moving")
            self.assertTrue(plan.write_input_images)
            self.assertEqual(plan.single_output_dir, tmp_path / "outputs" / "single_run")
            self.assertEqual(plan.images["fixed"].image, tmp_path / "fixed.nii.gz")
            self.assertEqual(plan.images["moving"].image, tmp_path / "moving.nii.gz")
            self.assertEqual(
                plan.images["moving"].segmentations["brain_mask"],
                tmp_path / "moving_mask.nii.gz",
            )
            self.assertEqual(
                plan.images["moving"].space.resolution_um,
                (20.0, 20.0, 20.0),
            )
            self.assertEqual(plan.pairs[0].fixed_image_id, "fixed")
            self.assertEqual(plan.pairs[0].moving_image_id, "moving")

    def test_build_jobs_from_single_plan_applies_run_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "single.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    registration_presets = ["baseline_syn_kimlab"]
                    write_input_images = true
                    output_dir = "outputs/single_run"

                    [image_defaults]
                    resolution_um = 20.0

                    [images.fixed]
                    image = "fixed.nii.gz"
                    orientation = "las"

                    [images.moving]
                    image = "moving.nii.gz"
                    orientation = "lsp"

                    [single]
                    fixed_image = "fixed"
                    moving_image = "moving"
                    """
                ).strip(),
                encoding="utf-8",
            )

            plan = load_registration_plan(config_path)
            jobs = build_jobs_from_plan(plan)

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].output_dir, tmp_path / "outputs" / "single_run")
            self.assertTrue(jobs[0].parameters.execution.write_input_images)
            self.assertEqual(jobs[0].parameters.name, "baseline_syn_kimlab")

    def test_batch_plan_resolves_template_role_and_segmentations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "batch.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    registration_presets = ["tuned_syn_cc"]
                    output_root = "outputs/batch"

                    [image_defaults]
                    resolution_um = 20.0

                    [images.subject_a]
                    image = "subject_a.nii.gz"
                    orientation = "las"

                    [images.template_p56]
                    image = "template_p56.nii.gz"
                    orientation = "lsp"

                    [images.template_p56.segmentations]
                    annotation = "template_annotation.nii.gz"

                    [batch]
                    template_role = "moving"
                    image_to_template = { subject_a = "template_p56" }
                    """
                ).strip(),
                encoding="utf-8",
            )

            plan = load_registration_plan(config_path)

            self.assertEqual(plan.mode, "batch")
            self.assertEqual(plan.output_root, tmp_path / "outputs" / "batch")
            self.assertEqual(plan.pairs[0].fixed_image_id, "subject_a")
            self.assertEqual(plan.pairs[0].moving_image_id, "template_p56")
            self.assertEqual(
                plan.images["template_p56"].segmentations["annotation"],
                tmp_path / "template_annotation.nii.gz",
            )

    def test_sweep_jobs_use_pair_and_preset_output_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "sweep.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    registration_presets = ["baseline_syn_kimlab", "tuned_syn_cc"]
                    output_root = "outputs/sweep"

                    [image_defaults]
                    resolution_um = 20.0

                    [images.shared]
                    image = "shared.nii.gz"
                    orientation = "lsp"

                    [images.run_image]
                    image = "run_image.nii.gz"
                    orientation = "las"

                    [sweep]
                    shared_image = "shared"
                    shared_image_role = "fixed"
                    run_images = ["run_image"]
                    """
                ).strip(),
                encoding="utf-8",
            )

            plan = load_registration_plan(config_path)
            jobs = build_jobs_from_plan(plan)

            self.assertEqual(len(jobs), 2)
            output_dirs = {job.output_dir for job in jobs}
            self.assertEqual(
                output_dirs,
                {
                    tmp_path
                    / "outputs"
                    / "sweep"
                    / "shared__run_image"
                    / "baseline_syn_kimlab",
                    tmp_path
                    / "outputs"
                    / "sweep"
                    / "shared__run_image"
                    / "tuned_syn_cc",
                },
            )

    def test_job_spec_rejects_multiple_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "invalid.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [run]
                    registration_presets = ["baseline_syn_kimlab"]
                    output_dir = "outputs/single_run"

                    [image_defaults]
                    resolution_um = 20.0

                    [images.fixed]
                    image = "fixed.nii.gz"
                    orientation = "las"

                    [images.moving]
                    image = "moving.nii.gz"
                    orientation = "lsp"

                    [single]
                    fixed_image = "fixed"
                    moving_image = "moving"

                    [batch]
                    template_role = "moving"
                    image_to_template = { fixed = "moving" }
                    """
                ).strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "Exactly one of \\[single\\], \\[batch\\], or \\[sweep\\]",
            ):
                load_registration_plan(config_path)


if __name__ == "__main__":
    unittest.main()
