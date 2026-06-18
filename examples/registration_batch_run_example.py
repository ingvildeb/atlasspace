from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from atlasspace.config.config_loading import (  # noqa: E402
    load_registration_batch_config,
    load_registration_parameters_config,
)
from atlasspace.registration.antspy_registration import run_antspy_registration  # noqa: E402
from atlasspace.registration.job_building import build_batch_jobs  # noqa: E402


EXAMPLE_BATCH_CONFIG = REPO_ROOT / "examples" / "configs" / "registration_batch_example.yaml"


def main() -> None:
    batch_config = load_registration_batch_config(EXAMPLE_BATCH_CONFIG)
    preset = load_registration_parameters_config(batch_config.registration_preset)
    jobs = build_batch_jobs(batch_config, preset)
    first_job = jobs[0]

    result = run_antspy_registration(first_job)

    print("Batch example finished.")
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
