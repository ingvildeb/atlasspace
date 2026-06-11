from __future__ import annotations

import sys
from pathlib import Path

# Update this path to your local atlasbuilder checkout if needed.
REPO_ROOT = Path(r"C:\Users\SmartBrain_32C_TR\Documents\GitHub\atlasbuilder")
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from atlasbuilder.config.config_loading import (  # noqa: E402
    load_registration_parameters_config,
    load_registration_sweep_config,
)
from atlasbuilder.registration.antspy_registration import run_antspy_registration  # noqa: E402
from atlasbuilder.registration.job_building import build_sweep_jobs  # noqa: E402


EXAMPLE_SWEEP_CONFIG = (
    REPO_ROOT
    / "examples"
    / "configs"
    / "registration_sweep_template_vs_ccfv3.yaml"
)


def main() -> None:
    sweep_config = load_registration_sweep_config(EXAMPLE_SWEEP_CONFIG)
    parameter_configs = [
        load_registration_parameters_config(preset_path)
        for preset_path in sweep_config.registration_presets
    ]
    jobs = build_sweep_jobs(sweep_config, parameter_configs)

    print(f"Prepared {len(jobs)} sweep jobs.")
    for index, job in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] "
            f"fixed={job.fixed_image_config.image_id} "
            f"moving={job.moving_image_config.image_id} "
            f"preset={job.parameters.name}"
        )
        result = run_antspy_registration(job)
        print(
            f"  success={result.success} "
            f"runtime_seconds={result.runtime_seconds} "
            f"output_dir={result.output_dir}"
        )
        if result.error_message is not None:
            print(f"  error={result.error_message}")


if __name__ == "__main__":
    main()
