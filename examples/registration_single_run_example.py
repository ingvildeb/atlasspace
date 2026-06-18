from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from atlasspace.config.config_models import ImageConfig
from atlasspace.config.space_models import SpaceDefinition
from atlasspace import registration


# Test data are intentionally kept outside the repo because the NIfTI files can
# be large and should not be versioned with atlasspace itself.
TEST_DATA_DIR = Path(
    r"Z:\path\to\atlasspace_test_data\registration_example"
)
OUTPUT_DIR = TEST_DATA_DIR / "outputs" / "single_run_baseline"


def main() -> None:
    preset = registration.load_preset("baseline_syn_kimlab")

    fixed_image = ImageConfig(
        image_id="fixed_template",
        image=TEST_DATA_DIR / "100478_ch1_iso20um.nii.gz",
        space=SpaceDefinition(
            space_name="subject_space",
            orientation="las",
            resolution_um=(20.0, 20.0, 20.0),
        ),
    )

    moving_image = ImageConfig(
        image_id="moving_template",
        image=TEST_DATA_DIR / "moving_preprocessed.nii.gz",
        space=SpaceDefinition(
            space_name="template_space",
            orientation="lsp",
            resolution_um=(20.0, 20.0, 20.0),
        ),
    )

    job = registration.RegistrationJob(
        fixed_image_config=fixed_image,
        moving_image_config=moving_image,
        output_dir=OUTPUT_DIR,
        parameters=preset,
        orientation_alignment="fixed_to_moving",
    )

    result = registration.run_antspy_registration(job)

    print("Registration finished.")
    print(f"Success: {result.success}")
    print(f"Fixed image id: {result.fixed_image_id}")
    print(f"Moving image id: {result.moving_image_id}")
    print(f"Preset: {result.preset_name}")
    print(f"Output directory: {result.output_dir}")
    print(f"Runtime seconds: {result.runtime_seconds}")
    print(f"Warped image: {result.warped_image}")
    print(f"Inverse warped image: {result.inverse_warped_image}")
    print(f"Forward transforms: {result.forward_transforms}")
    print(f"Inverse transforms: {result.inverse_transforms}")
    print(f"Error message: {result.error_message}")


if __name__ == "__main__":
    main()
