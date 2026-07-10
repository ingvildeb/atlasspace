from __future__ import annotations

from pathlib import Path

from atlasspace.config.job_spec_models import RegistrationPair, RegistrationPlan
from atlasspace.config.preset_models import RegistrationParametersConfig
from atlasspace.config.config_loading import load_preset
from atlasspace.runtime.registration import RegistrationJob


def build_jobs_from_plan(plan: RegistrationPlan) -> list[RegistrationJob]:
    parameter_configs = [
        _apply_run_overrides(
            load_preset(preset_reference),
            write_input_images=plan.write_input_images,
        )
        for preset_reference in plan.preset_references
    ]
    _validate_unique_preset_names(parameter_configs)

    jobs: list[RegistrationJob] = []
    for pair in plan.pairs:
        fixed_image_config = plan.images[pair.fixed_image_id]
        moving_image_config = plan.images[pair.moving_image_id]
        for parameters_config in parameter_configs:
            jobs.append(
                RegistrationJob(
                    fixed_image_config=fixed_image_config,
                    moving_image_config=moving_image_config,
                    output_dir=_derive_plan_output_dir(
                        plan,
                        pair=pair,
                        parameters_config=parameters_config,
                    ),
                    parameters=parameters_config,
                    orientation_alignment=plan.orientation_alignment,
                    moving_segmentation_policy=plan.moving_segmentations,
                )
            )
    _validate_unique_output_dirs(jobs)
    return jobs


def _derive_plan_output_dir(
    plan: RegistrationPlan,
    *,
    pair: RegistrationPair,
    parameters_config: RegistrationParametersConfig,
) -> Path:
    if plan.mode == "single":
        if plan.single_output_dir is None:
            raise ValueError("single_output_dir must be provided for single plans.")
        return plan.single_output_dir

    if plan.mode == "batch":
        if plan.batch_output_subdir is None:
            raise ValueError("batch_output_subdir must be provided for batch plans.")
        if pair.run_image_id is None:
            raise ValueError("run_image_id must be provided for batch plans.")
        run_image_config = plan.images.get(pair.run_image_id)
        if run_image_config is None:
            raise ValueError(
                f"run_image_id '{pair.run_image_id}' was not found in plan images."
            )
        return run_image_config.image.parent / Path(plan.batch_output_subdir)

    if plan.output_root is None:
        raise ValueError("output_root must be provided for sweep plans.")
    return plan.output_root / pair.pair_id / parameters_config.name


def _apply_run_overrides(
    parameters_config: RegistrationParametersConfig,
    *,
    write_input_images: bool,
) -> RegistrationParametersConfig:
    parameters_copy = parameters_config.model_copy(deep=True)
    parameters_copy.execution.write_input_images = write_input_images
    return parameters_copy


def _validate_unique_preset_names(
    parameter_configs: list[RegistrationParametersConfig],
) -> None:
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
            "Preset names must be unique when building registration jobs. "
            f"Duplicate names found: {duplicate_preset_names}"
        )


def _validate_unique_output_dirs(jobs: list[RegistrationJob]) -> None:
    output_dirs = [job.output_dir for job in jobs]
    duplicate_output_dirs = sorted(
        {
            str(output_dir)
            for output_dir in output_dirs
            if output_dirs.count(output_dir) > 1
        }
    )
    if duplicate_output_dirs:
        raise ValueError(
            "Registration jobs resolved to duplicate output directories. "
            "Each job must have a unique output path. "
            f"Duplicate output directories: {duplicate_output_dirs}"
        )
