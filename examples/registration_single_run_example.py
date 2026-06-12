from __future__ import annotations

import sys
from pathlib import Path

# Update these paths to point at your local atlasbuilder checkout and test data.
REPO_ROOT = Path(r"C:\Users\SmartBrain_32C_TR\Documents\GitHub\atlasbuilder")
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from atlasbuilder.config.config_loading import load_registration_parameters_config
from atlasbuilder.config.config_models import ImageConfig
from atlasbuilder.config.space_models import SpaceDefinition
from atlasbuilder.registration.antspy_registration import run_antspy_registration
from atlasbuilder.runtime.registration import RegistrationJob


# Test data are intentionally kept outside the repo because the NIfTI files can
# be large and should not be versioned with atlasbuilder itself.
TEST_DATA_DIR = Path(
    r"Z:\Labmembers\Ingvild\standard_test_data\atlasbuilder\registration\with_reorienting"
)
OUTPUT_DIR = TEST_DATA_DIR / "outputs" / "single_run_baseline"


def main() -> None:
    preset = load_registration_parameters_config(
        REPO_ROOT
        / "configs"
        / "registration_presets"
        / "baseline_syn_kimlab.yaml"
    )

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

    job = RegistrationJob(
        fixed_image_config=fixed_image,
        moving_image_config=moving_image,
        output_dir=OUTPUT_DIR,
        parameters=preset,
        orientation_alignment="fixed_to_moving",
    )

    result = run_antspy_registration(job)

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
