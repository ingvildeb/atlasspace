from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from atlasspace.config.space_models import SpaceDefinition


IntensityNormalizationMode = Literal["zscore", "robust_zscore"]
RegistrationTransformType = Literal["Rigid", "Affine", "SyN", "SyNOnly"]
LinearMetricType = Literal["mattes"]
DeformableMetricType = Literal["mattes", "CC"]
SharedImageRole = Literal["fixed", "moving"]
OrientationAlignmentMode = Literal["none", "moving_to_fixed", "fixed_to_moving"]


class PreprocessingConfig(BaseModel):
    intensity_normalization: IntensityNormalizationMode | None = None
    minmax_clip_percentiles: tuple[float, float] | None = None
    histogram_match: bool = False
    gaussian_sigma_vox: float | None = None

    @model_validator(mode="after")
    def validate_preprocessing(self) -> "PreprocessingConfig":
        if self.minmax_clip_percentiles is not None:
            low, high = self.minmax_clip_percentiles
            if not (0 <= low < high <= 100):
                raise ValueError(
                    "minmax_clip_percentiles must satisfy 0 <= low < high <= 100."
                )
        if self.gaussian_sigma_vox is not None and self.gaussian_sigma_vox <= 0:
            raise ValueError("gaussian_sigma_vox must be positive when provided.")
        return self


class RegistrationSettingsConfig(BaseModel):
    working_resolution_um: int = 20
    transform_type: RegistrationTransformType = "SyN"

    aff_metric: LinearMetricType = "mattes"
    aff_sampling: int = 32
    aff_random_sampling_rate: float = 0.25
    aff_iterations: tuple[int, ...] = (1000, 1000, 1000)
    aff_shrink_factors: tuple[int, ...] = (12, 8, 4)
    aff_smoothing_sigmas: tuple[int, ...] = (4, 3, 2)

    syn_metric: DeformableMetricType = "mattes"
    syn_sampling: int = 32
    syn_gradient_step: float = 0.1
    syn_flow_sigma: float = 3.0
    syn_total_sigma: float = 0.0
    syn_reg_iterations: tuple[int, ...] = (1000, 1000, 1000)

    @model_validator(mode="after")
    def validate_registration(self) -> "RegistrationSettingsConfig":
        if self.working_resolution_um <= 0:
            raise ValueError("working_resolution_um must be positive.")
        if self.aff_sampling <= 0:
            raise ValueError("aff_sampling must be positive.")
        if not (0 <= self.aff_random_sampling_rate <= 1):
            raise ValueError("aff_random_sampling_rate must be between 0 and 1.")
        if not self.aff_iterations or any(value < 0 for value in self.aff_iterations):
            raise ValueError("aff_iterations must be a non-empty sequence of nonnegative integers.")
        if not self.aff_shrink_factors or any(value <= 0 for value in self.aff_shrink_factors):
            raise ValueError("aff_shrink_factors must be a non-empty sequence of positive integers.")
        if not self.aff_smoothing_sigmas or any(value < 0 for value in self.aff_smoothing_sigmas):
            raise ValueError("aff_smoothing_sigmas must be a non-empty sequence of nonnegative numbers.")
        if self.syn_sampling <= 0:
            raise ValueError("syn_sampling must be positive.")
        if self.syn_gradient_step <= 0:
            raise ValueError("syn_gradient_step must be positive.")
        if self.syn_flow_sigma < 0:
            raise ValueError("syn_flow_sigma must be nonnegative.")
        if self.syn_total_sigma < 0:
            raise ValueError("syn_total_sigma must be nonnegative.")
        if not self.syn_reg_iterations or any(value < 0 for value in self.syn_reg_iterations):
            raise ValueError("syn_reg_iterations must be a non-empty sequence of nonnegative integers.")
        return self


class ExecutionConfig(BaseModel):
    threads: int = 0
    singleprecision: bool = True
    use_legacy_histogram_matching: bool = False
    verbose: bool = True
    random_seed: int | None = None
    write_input_images: bool = False

    @model_validator(mode="after")
    def validate_execution(self) -> "ExecutionConfig":
        if self.threads < 0:
            raise ValueError("threads must be zero or a positive integer.")
        if self.random_seed is not None and self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative when provided.")
        return self


class RegistrationParametersConfig(BaseModel):
    name: str
    description: str | None = None
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    registration: RegistrationSettingsConfig = Field(default_factory=RegistrationSettingsConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="after")
    def validate_name(self) -> "RegistrationParametersConfig":
        if not self.name.strip():
            raise ValueError("name must not be empty.")
        return self


class ImageConfig(BaseModel):
    image_id: str
    image: Path
    space: SpaceDefinition

    @model_validator(mode="after")
    def validate_image(self) -> "ImageConfig":
        if not self.image_id.strip():
            raise ValueError("image_id must not be empty.")
        return self


class RegistrationBatchConfig(BaseModel):
    shared_image_role: SharedImageRole
    shared_image: ImageConfig
    orientation_alignment: OrientationAlignmentMode = "none"
    registration_preset: Path
    output_subdir_name: str | None = None
    run_images: list[ImageConfig]

    @model_validator(mode="after")
    def validate_batch(self) -> "RegistrationBatchConfig":
        if not self.run_images:
            raise ValueError("run_images must not be empty.")
        if self.output_subdir_name is not None and not self.output_subdir_name.strip():
            raise ValueError("output_subdir_name must not be empty when provided.")
        return self


class RegistrationSweepConfig(BaseModel):
    shared_image_role: SharedImageRole
    shared_image: ImageConfig
    orientation_alignment: OrientationAlignmentMode = "none"
    registration_presets: list[Path]
    output_root: Path
    run_images: list[ImageConfig]

    @model_validator(mode="after")
    def validate_sweep(self) -> "RegistrationSweepConfig":
        if not self.registration_presets:
            raise ValueError("registration_presets must not be empty.")
        if not self.run_images:
            raise ValueError("run_images must not be empty.")
        return self
