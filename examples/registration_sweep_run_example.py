from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from atlasspace.config.config_loading import load_registration_plan  # noqa: E402
from atlasspace.config_templates import (  # noqa: E402
    REGISTRATION_SWEEP_TEMPLATE,
    as_template_path,
)
from atlasspace.registration.antspy_registration import run_antspy_registration  # noqa: E402
from atlasspace.registration.job_building import build_jobs_from_plan  # noqa: E402


def main() -> None:
    with as_template_path(REGISTRATION_SWEEP_TEMPLATE) as config_path:
        plan = load_registration_plan(config_path)
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
