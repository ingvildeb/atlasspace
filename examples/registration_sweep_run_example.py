from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from atlasspace.config.config_loading import load_registration_plan  # noqa: E402
from atlasspace.registration.antspy_registration import run_antspy_registration  # noqa: E402
from atlasspace.registration.job_building import build_jobs_from_plan  # noqa: E402


EXAMPLE_SWEEP_CONFIG = (
    REPO_ROOT
    / "examples"
    / "configs"
    / "registration_sweep_template.toml"
)


def main() -> None:
    plan = load_registration_plan(EXAMPLE_SWEEP_CONFIG)
    jobs = build_jobs_from_plan(plan)

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
