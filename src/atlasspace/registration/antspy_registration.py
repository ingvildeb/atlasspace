from __future__ import annotations

import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

try:
    import ants
except ImportError as exc:
    raise ImportError(
        "atlasspace.registration.antspy_registration requires antspyx. "
        "Install atlasspace with `pip install -e .` or otherwise ensure antspyx is available."
    ) from exc

from atlasspace.config.image_models import ImageConfig
from atlasspace.config.preset_models import RegistrationParametersConfig
from atlasspace.config.space_models import SpaceDefinition
from atlasspace.image.reorientation import reorient_array_to_match, spaces_match_orientation
from atlasspace.image.space_validation import validate_isotropic_space
from atlasspace.io.nifti import load_nifti_array, write_nifti_from_array
from atlasspace.registration.preprocessing import (
    preprocess_registration_images,
    resample_to_resolution,
)
from atlasspace.registration.result_manifest import write_registration_result_manifest
from atlasspace.runtime.registration import RegistrationJob, RegistrationResult
from atlasspace.runtime.transforms import TransformSequence
from atlasspace.transforms.application import transform_segmentation


def build_antspy_registration_kwargs(
    parameters: RegistrationParametersConfig,
) -> dict[str, object]:
    registration = parameters.registration
    execution = parameters.execution

    return {
        "type_of_transform": registration.transform_type,
        "aff_metric": registration.aff_metric,
        "aff_sampling": registration.aff_sampling,
        "aff_random_sampling_rate": registration.aff_random_sampling_rate,
        "aff_iterations": registration.aff_iterations,
        "aff_shrink_factors": registration.aff_shrink_factors,
        "aff_smoothing_sigmas": registration.aff_smoothing_sigmas,
        "syn_metric": registration.syn_metric,
        "syn_sampling": registration.syn_sampling,
        "grad_step": registration.syn_gradient_step,
        "flow_sigma": registration.syn_flow_sigma,
        "total_sigma": registration.syn_total_sigma,
        "reg_iterations": registration.syn_reg_iterations,
        "singleprecision": execution.singleprecision,
        "use_legacy_histogram_matching": execution.use_legacy_histogram_matching,
        "verbose": execution.verbose,
    }


@contextmanager
def _thread_environment(threads: int):
    env_key = "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
    previous_value = os.environ.get(env_key)
    if threads > 0:
        os.environ[env_key] = str(threads)
    try:
        yield
    finally:
        if threads > 0:
            if previous_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = previous_value


def _save_input_images(job: RegistrationJob, output_dir: Path) -> tuple[Path, Path]:
    fixed_input_path = output_dir / "fixed_input.nii.gz"
    moving_input_path = output_dir / "moving_input.nii.gz"
    shutil.copy2(job.fixed_image_config.image, fixed_input_path)
    shutil.copy2(job.moving_image_config.image, moving_input_path)
    return fixed_input_path, moving_input_path


def prepare_image_for_registration(
    input_path: Path,
    input_space: SpaceDefinition,
    output_path: Path,
    target_space: SpaceDefinition | None = None,
) -> tuple[Path, SpaceDefinition]:
    validate_isotropic_space(input_space)

    source_array = load_nifti_array(input_path)
    effective_array = source_array
    effective_space = input_space

    if target_space is not None and not spaces_match_orientation(input_space, target_space):
        effective_array, effective_space = reorient_array_to_match(
            source_array,
            input_space,
            target_space,
        )

    normalized_path = write_nifti_from_array(
        effective_array,
        effective_space,
        output_path,
    )
    return normalized_path, effective_space


def _prepare_registration_inputs(
    job: RegistrationJob,
) -> tuple[Path, Path, SpaceDefinition, SpaceDefinition]:
    fixed_space = job.fixed_image_config.space
    moving_space = job.moving_image_config.space
    alignment_mode = job.orientation_alignment

    fixed_target_space: SpaceDefinition | None = None
    moving_target_space: SpaceDefinition | None = None

    if alignment_mode == "moving_to_fixed":
        moving_target_space = fixed_space
    elif alignment_mode == "fixed_to_moving":
        fixed_target_space = moving_space
    elif alignment_mode != "none":
        raise ValueError(f"Unsupported orientation_alignment mode: {alignment_mode}")

    fixed_path, effective_fixed_space = prepare_image_for_registration(
        job.fixed_image_config.image,
        fixed_space,
        job.output_dir / "fixed_normalized_for_registration.nii.gz",
        target_space=fixed_target_space,
    )
    moving_path, effective_moving_space = prepare_image_for_registration(
        job.moving_image_config.image,
        moving_space,
        job.output_dir / "moving_normalized_for_registration.nii.gz",
        target_space=moving_target_space,
    )
    return fixed_path, moving_path, effective_fixed_space, effective_moving_space


def _load_prepared_images(
    fixed_path: Path,
    moving_path: Path,
) -> tuple[ants.ANTsImage, ants.ANTsImage]:
    fixed_native = ants.image_read(str(fixed_path))
    moving_native = ants.image_read(str(moving_path))
    return fixed_native, moving_native


def _build_registration_space_images(
    parameters: RegistrationParametersConfig,
    fixed_native: ants.ANTsImage,
    moving_native: ants.ANTsImage,
    fixed_space: SpaceDefinition,
    moving_space: SpaceDefinition,
) -> tuple[ants.ANTsImage, ants.ANTsImage]:
    fixed_preprocessed, moving_preprocessed = preprocess_registration_images(
        fixed_native,
        moving_native,
        parameters.preprocessing,
    )

    working_resolution_um = parameters.registration.working_resolution_um
    fixed_registration = resample_to_resolution(
        fixed_preprocessed,
        nominal_resolution_um=fixed_space.resolution_um[0],
        target_resolution_um=working_resolution_um,
    )
    moving_registration = resample_to_resolution(
        moving_preprocessed,
        nominal_resolution_um=moving_space.resolution_um[0],
        target_resolution_um=working_resolution_um,
    )
    return fixed_registration, moving_registration


def _normalize_transform_paths(
    transform_value: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    if transform_value is None:
        return []
    if isinstance(transform_value, str):
        return [transform_value]
    return list(transform_value)


def _apply_transforms_to_native_resolution(
    fixed_native: ants.ANTsImage,
    moving_native: ants.ANTsImage,
    antspy_result: dict[str, object],
) -> tuple[ants.ANTsImage | None, ants.ANTsImage | None, list[str], list[str]]:
    forward_transforms = _normalize_transform_paths(antspy_result.get("fwdtransforms"))
    inverse_transforms = _normalize_transform_paths(antspy_result.get("invtransforms"))

    warped_native: ants.ANTsImage | None = None
    inverse_warped_native: ants.ANTsImage | None = None

    if forward_transforms:
        warped_native = ants.apply_transforms(
            fixed=fixed_native,
            moving=moving_native,
            transformlist=forward_transforms,
        )
    elif antspy_result.get("warpedmovout") is not None:
        warped_native = antspy_result["warpedmovout"]

    if inverse_transforms:
        inverse_warped_native = ants.apply_transforms(
            fixed=moving_native,
            moving=fixed_native,
            transformlist=inverse_transforms,
        )
    elif antspy_result.get("warpedfixout") is not None:
        inverse_warped_native = antspy_result["warpedfixout"]

    return warped_native, inverse_warped_native, forward_transforms, inverse_transforms


def _save_registration_outputs(
    output_dir: Path,
    warped_native: ants.ANTsImage | None,
    inverse_warped_native: ants.ANTsImage | None,
    fixed_output_space: SpaceDefinition,
    moving_output_space: SpaceDefinition,
    forward_transforms: list[str],
    inverse_transforms: list[str],
) -> tuple[Path | None, Path | None, list[Path], list[Path]]:
    warped_path: Path | None = None
    inverse_warped_path: Path | None = None

    if warped_native is not None:
        warped_path = output_dir / "ANTsPy_Warped.nii.gz"
        write_nifti_from_array(
            warped_native.numpy(),
            fixed_output_space,
            warped_path,
        )

    if inverse_warped_native is not None:
        inverse_warped_path = output_dir / "ANTsPy_InverseWarped.nii.gz"
        write_nifti_from_array(
            inverse_warped_native.numpy(),
            moving_output_space,
            inverse_warped_path,
        )

    forward_transform_paths = [Path(path) for path in forward_transforms]
    inverse_transform_paths = [Path(path) for path in inverse_transforms]
    return (
        warped_path,
        inverse_warped_path,
        forward_transform_paths,
        inverse_transform_paths,
    )


def _write_parameters_snapshot(
    job: RegistrationJob,
    output_dir: Path,
) -> Path:
    snapshot_path = output_dir / "registration_parameters.yaml"
    snapshot_path.write_text(
        yaml.safe_dump(
            job.parameters.model_dump(mode="python"),
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return snapshot_path


def _moving_segmentation_output_dir(job: RegistrationJob) -> Path:
    output_dir = job.output_dir
    policy = job.moving_segmentation_policy
    if policy.output_subdir is not None:
        output_dir = output_dir / policy.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _transform_moving_segmentations(
    job: RegistrationJob,
    result: RegistrationResult,
) -> dict[str, Path]:
    policy = job.moving_segmentation_policy
    if not policy.enabled or not job.moving_image_config.segmentations:
        return {}

    output_dir = _moving_segmentation_output_dir(job)
    transform_sequence = TransformSequence.from_registration_result(result)
    transformed_segmentations: dict[str, Path] = {}

    for segmentation_name, segmentation_path in job.moving_image_config.segmentations.items():
        segmentation_config = ImageConfig(
            image_id=segmentation_name,
            image=segmentation_path,
            space=job.moving_image_config.space,
        )
        output_path = output_dir / f"{segmentation_name}_WarpedSegmentation.nii.gz"
        transformed_config = transform_segmentation(
            segmentation_config,
            transform_sequence,
            job.fixed_image_config,
            direction="forward",
            interpolation=policy.interpolation,
            output_path=output_path,
            write_intermediates=policy.write_intermediates,
        )
        transformed_segmentations[segmentation_name] = transformed_config.image

    return transformed_segmentations


def run_antspy_registration(job: RegistrationJob) -> RegistrationResult:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    fixed_normalized_path = job.output_dir / "fixed_normalized_for_registration.nii.gz"
    moving_normalized_path = job.output_dir / "moving_normalized_for_registration.nii.gz"

    try:
        if job.parameters.execution.write_input_images:
            _save_input_images(job, job.output_dir)

        (
            fixed_normalized_path,
            moving_normalized_path,
            fixed_space,
            moving_space,
        ) = _prepare_registration_inputs(job)

        fixed_native, moving_native = _load_prepared_images(
            fixed_normalized_path,
            moving_normalized_path,
        )

        fixed_registration, moving_registration = _build_registration_space_images(
            job.parameters,
            fixed_native,
            moving_native,
            fixed_space,
            moving_space,
        )

        registration_kwargs = build_antspy_registration_kwargs(job.parameters)
        outprefix = str(job.output_dir / "ANTsPy_")

        with _thread_environment(job.parameters.execution.threads):
            antspy_result = ants.registration(
                fixed=fixed_registration,
                moving=moving_registration,
                outprefix=outprefix,
                **registration_kwargs,
            )

        (
            warped_native,
            inverse_warped_native,
            forward_transforms,
            inverse_transforms,
        ) = _apply_transforms_to_native_resolution(
            fixed_native,
            moving_native,
            antspy_result,
        )

        (
            warped_path,
            inverse_warped_path,
            forward_transform_paths,
            inverse_transform_paths,
        ) = _save_registration_outputs(
            job.output_dir,
            warped_native,
            inverse_warped_native,
            fixed_space,
            moving_space,
            forward_transforms,
            inverse_transforms,
        )

        runtime_seconds = time.perf_counter() - start_time
        result = RegistrationResult(
            fixed_image_id=job.fixed_image_config.image_id,
            moving_image_id=job.moving_image_config.image_id,
            preset_name=job.parameters.name,
            output_dir=job.output_dir,
            success=True,
            declared_fixed_space=job.fixed_image_config.space,
            declared_moving_space=job.moving_image_config.space,
            effective_fixed_space=fixed_space,
            effective_moving_space=moving_space,
            runtime_seconds=runtime_seconds,
            warped_image=warped_path,
            inverse_warped_image=inverse_warped_path,
            forward_transforms=forward_transform_paths,
            inverse_transforms=inverse_transform_paths,
            transformed_segmentations={},
            error_message=None,
        )
        result.transformed_segmentations = _transform_moving_segmentations(job, result)
        _write_parameters_snapshot(job, job.output_dir)
        write_registration_result_manifest(
            job,
            result,
            fixed_normalized_path=fixed_normalized_path,
            moving_normalized_path=moving_normalized_path,
        )
        return result
    except Exception as exc:
        runtime_seconds = time.perf_counter() - start_time
        fixed_space = job.fixed_image_config.space
        moving_space = job.moving_image_config.space
        result = RegistrationResult(
            fixed_image_id=job.fixed_image_config.image_id,
            moving_image_id=job.moving_image_config.image_id,
            preset_name=job.parameters.name,
            output_dir=job.output_dir,
            success=False,
            declared_fixed_space=job.fixed_image_config.space,
            declared_moving_space=job.moving_image_config.space,
            effective_fixed_space=fixed_space,
            effective_moving_space=moving_space,
            runtime_seconds=runtime_seconds,
            warped_image=None,
            inverse_warped_image=None,
            forward_transforms=[],
            inverse_transforms=[],
            transformed_segmentations={},
            error_message=str(exc),
        )
        _write_parameters_snapshot(job, job.output_dir)
        write_registration_result_manifest(
            job,
            result,
            fixed_normalized_path=(
                fixed_normalized_path if fixed_normalized_path.exists() else None
            ),
            moving_normalized_path=(
                moving_normalized_path if moving_normalized_path.exists() else None
            ),
        )
        return result
