from __future__ import annotations

from pathlib import Path

from atlasspace.config.config_models import (
    ImageConfig,
    RegistrationBatchConfig,
    RegistrationParametersConfig,
    RegistrationSweepConfig,
    SharedImageRole,
)
from atlasspace.runtime.registration import RegistrationJob


def derive_batch_output_dir(
    run_image_config: ImageConfig,
    output_subdir_name: str | None,
) -> Path:
    if output_subdir_name is None:
        raise ValueError("output_subdir_name must be provided for batch job construction.")
    return run_image_config.image.parent / output_subdir_name


def resolve_registration_pair(
    shared_image_role: SharedImageRole,
    shared_image_config: ImageConfig,
    run_image_config: ImageConfig,
) -> tuple[ImageConfig, ImageConfig]:
    if shared_image_role == "fixed":
        return shared_image_config, run_image_config
    return run_image_config, shared_image_config


def build_batch_jobs(
    batch_config: RegistrationBatchConfig,
    parameters_config: RegistrationParametersConfig,
) -> list[RegistrationJob]:
    jobs: list[RegistrationJob] = []
    for run_image_config in batch_config.run_images:
        fixed_image_config, moving_image_config = resolve_registration_pair(
            batch_config.shared_image_role,
            batch_config.shared_image,
            run_image_config,
        )
        jobs.append(
            RegistrationJob(
                fixed_image_config=fixed_image_config,
                moving_image_config=moving_image_config,
                output_dir=derive_batch_output_dir(
                    run_image_config,
                    batch_config.output_subdir_name,
                ),
                parameters=parameters_config,
                orientation_alignment=batch_config.orientation_alignment,
            )
        )
    return jobs


def build_sweep_jobs(
    sweep_config: RegistrationSweepConfig,
    parameter_configs: list[RegistrationParametersConfig],
) -> list[RegistrationJob]:
    if not parameter_configs:
        raise ValueError("parameter_configs must not be empty.")

    preset_names = [config.name for config in parameter_configs]
    duplicate_preset_names = sorted(
        {
            preset_name
            for preset_name in preset_names
            if preset_names.count(preset_name) > 1
        }
    )
    if duplicate_preset_names:
        raise ValueError(
            "Preset names must be unique when building sweep jobs. "
            f"Duplicate names found: {duplicate_preset_names}"
        )

    jobs: list[RegistrationJob] = []
    for run_image_config in sweep_config.run_images:
        for parameters_config in parameter_configs:
            fixed_image_config, moving_image_config = resolve_registration_pair(
                sweep_config.shared_image_role,
                sweep_config.shared_image,
                run_image_config,
            )
            jobs.append(
                RegistrationJob(
                    fixed_image_config=fixed_image_config,
                    moving_image_config=moving_image_config,
                    output_dir=sweep_config.output_root
                    / f"{run_image_config.image_id}_{parameters_config.name}",
                    parameters=parameters_config,
                    orientation_alignment=sweep_config.orientation_alignment,
                )
            )
    return jobs
