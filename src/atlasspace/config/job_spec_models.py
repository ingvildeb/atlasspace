from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlasspace.config.image_models import (
    ImageConfig,
    ImageRole,
    OrientationAlignmentMode,
)


RegistrationMode = Literal["single", "batch", "sweep"]
MovingSegmentationInterpolationMode = Literal["genericLabel", "nearestNeighbor"]


class RegistrationRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registration_presets: list[str]
    orientation_alignment: OrientationAlignmentMode = "none"
    write_input_images: bool = False
    output_dir: Path | None = None
    output_subdir: str | None = None
    output_root: Path | None = None

    @model_validator(mode="after")
    def validate_run(self) -> "RegistrationRunConfig":
        if not self.registration_presets:
            raise ValueError("registration_presets must not be empty.")
        if any(not preset.strip() for preset in self.registration_presets):
            raise ValueError("registration_presets entries must not be empty.")
        if self.output_subdir is not None:
            if not self.output_subdir.strip():
                raise ValueError("output_subdir must not be empty when provided.")
            if Path(self.output_subdir).is_absolute():
                raise ValueError("output_subdir must be relative when provided.")
        return self


class ImageDefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orientation: str | None = None
    resolution_um: float | None = None

    @model_validator(mode="after")
    def validate_defaults(self) -> "ImageDefaultsConfig":
        if self.orientation is not None and not self.orientation.strip():
            raise ValueError("orientation must not be empty when provided.")
        if self.resolution_um is not None and self.resolution_um <= 0:
            raise ValueError("resolution_um must be positive when provided.")
        return self


class MovingSegmentationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    interpolation: MovingSegmentationInterpolationMode = "genericLabel"
    output_subdir: str | None = None
    write_intermediates: bool = False

    @model_validator(mode="after")
    def validate_policy(self) -> "MovingSegmentationPolicy":
        if self.output_subdir is not None:
            if not self.output_subdir.strip():
                raise ValueError("output_subdir must not be empty when provided.")
            if Path(self.output_subdir).is_absolute():
                raise ValueError("output_subdir must be relative when provided.")
        return self


class RegistrationJobSpecImageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: Path
    space_name: str | None = None
    orientation: str | None = None
    resolution_um: float | None = None
    segmentations: dict[str, Path] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_image(self) -> "RegistrationJobSpecImageConfig":
        if self.space_name is not None and not self.space_name.strip():
            raise ValueError("space_name must not be empty when provided.")
        if self.orientation is not None and not self.orientation.strip():
            raise ValueError("orientation must not be empty when provided.")
        if self.resolution_um is not None and self.resolution_um <= 0:
            raise ValueError("resolution_um must be positive when provided.")
        if any(not segmentation_id.strip() for segmentation_id in self.segmentations):
            raise ValueError("segmentation ids must not be empty.")
        return self


class SingleRegistrationModeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixed_image: str
    moving_image: str

    @model_validator(mode="after")
    def validate_single(self) -> "SingleRegistrationModeConfig":
        if not self.fixed_image.strip():
            raise ValueError("fixed_image must not be empty.")
        if not self.moving_image.strip():
            raise ValueError("moving_image must not be empty.")
        return self


class BatchRegistrationModeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_role: ImageRole
    image_to_template: dict[str, str]

    @model_validator(mode="after")
    def validate_batch(self) -> "BatchRegistrationModeConfig":
        if not self.image_to_template:
            raise ValueError("image_to_template must not be empty.")
        if any(not image_id.strip() for image_id in self.image_to_template):
            raise ValueError("image_to_template keys must not be empty.")
        if any(not template_id.strip() for template_id in self.image_to_template.values()):
            raise ValueError("image_to_template values must not be empty.")
        return self


class SweepRegistrationModeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shared_image: str
    shared_image_role: ImageRole
    run_images: list[str]

    @model_validator(mode="after")
    def validate_sweep(self) -> "SweepRegistrationModeConfig":
        if not self.shared_image.strip():
            raise ValueError("shared_image must not be empty.")
        if not self.run_images:
            raise ValueError("run_images must not be empty.")
        if any(not image_id.strip() for image_id in self.run_images):
            raise ValueError("run_images entries must not be empty.")
        return self


class RegistrationPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixed_image_id: str
    moving_image_id: str
    run_image_id: str | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> "RegistrationPair":
        if not self.fixed_image_id.strip():
            raise ValueError("fixed_image_id must not be empty.")
        if not self.moving_image_id.strip():
            raise ValueError("moving_image_id must not be empty.")
        if self.fixed_image_id == self.moving_image_id:
            raise ValueError("fixed_image_id and moving_image_id must differ.")
        if self.run_image_id is not None and not self.run_image_id.strip():
            raise ValueError("run_image_id must not be empty when provided.")
        if (
            self.run_image_id is not None
            and self.run_image_id not in {self.fixed_image_id, self.moving_image_id}
        ):
            raise ValueError(
                "run_image_id must match either fixed_image_id or moving_image_id when provided."
            )
        return self

    @property
    def pair_id(self) -> str:
        return f"{self.fixed_image_id}__{self.moving_image_id}"


class RegistrationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: RegistrationMode
    preset_references: list[str]
    orientation_alignment: OrientationAlignmentMode = "none"
    write_input_images: bool = False
    single_output_dir: Path | None = None
    batch_output_subdir: str | None = None
    output_root: Path | None = None
    images: dict[str, ImageConfig]
    pairs: list[RegistrationPair]
    moving_segmentations: MovingSegmentationPolicy = Field(
        default_factory=MovingSegmentationPolicy
    )

    @model_validator(mode="after")
    def validate_plan(self) -> "RegistrationPlan":
        if not self.preset_references:
            raise ValueError("preset_references must not be empty.")
        if not self.images:
            raise ValueError("images must not be empty.")
        if not self.pairs:
            raise ValueError("pairs must not be empty.")
        if self.mode == "single":
            if self.single_output_dir is None:
                raise ValueError("single_output_dir is required for single plans.")
        elif self.mode == "batch":
            if self.batch_output_subdir is None:
                raise ValueError("batch_output_subdir is required for batch plans.")
        elif self.output_root is None:
            raise ValueError("output_root is required for sweep plans.")

        missing_ids = {
            image_id
            for pair in self.pairs
            for image_id in (pair.fixed_image_id, pair.moving_image_id)
            if image_id not in self.images
        }
        if missing_ids:
            raise ValueError(
                "pairs reference unknown image ids: "
                f"{sorted(missing_ids)}"
            )
        missing_run_ids = {
            pair.run_image_id
            for pair in self.pairs
            if pair.run_image_id is not None and pair.run_image_id not in self.images
        }
        if missing_run_ids:
            raise ValueError(
                "pairs reference unknown run image ids: "
                f"{sorted(missing_run_ids)}"
            )
        return self


class RegistrationJobSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RegistrationRunConfig
    image_defaults: ImageDefaultsConfig = Field(default_factory=ImageDefaultsConfig)
    moving_segmentations: MovingSegmentationPolicy = Field(
        default_factory=MovingSegmentationPolicy
    )
    images: dict[str, RegistrationJobSpecImageConfig]
    single: SingleRegistrationModeConfig | None = None
    batch: BatchRegistrationModeConfig | None = None
    sweep: SweepRegistrationModeConfig | None = None

    @model_validator(mode="after")
    def validate_job_spec(self) -> "RegistrationJobSpecConfig":
        if not self.images:
            raise ValueError("images must not be empty.")
        if any(not image_id.strip() for image_id in self.images):
            raise ValueError("image ids must not be empty.")

        active_modes = [
            mode_name
            for mode_name, mode_config in (
                ("single", self.single),
                ("batch", self.batch),
                ("sweep", self.sweep),
            )
            if mode_config is not None
        ]
        if len(active_modes) != 1:
            raise ValueError(
                "Exactly one of [single], [batch], or [sweep] must be provided."
            )

        mode = active_modes[0]
        if mode in {"single", "batch"} and len(self.run.registration_presets) != 1:
            raise ValueError(
                f"{mode} configs must define exactly 1 registration preset."
            )

        if mode == "single":
            if self.run.output_dir is None:
                raise ValueError("[run].output_dir is required for single configs.")
            if self.run.output_subdir is not None:
                raise ValueError("[run].output_subdir is not used for single configs.")
            if self.run.output_root is not None:
                raise ValueError("[run].output_root is not used for single configs.")
            single_config = self.single
            assert single_config is not None
            self._validate_image_reference(single_config.fixed_image, "single.fixed_image")
            self._validate_image_reference(
                single_config.moving_image,
                "single.moving_image",
            )
            if single_config.fixed_image == single_config.moving_image:
                raise ValueError("single.fixed_image and single.moving_image must differ.")
            return self

        if mode == "batch":
            if self.run.output_subdir is None:
                raise ValueError("[run].output_subdir is required for batch configs.")
            if self.run.output_dir is not None:
                raise ValueError("[run].output_dir is not used for batch configs.")
            if self.run.output_root is not None:
                raise ValueError("[run].output_root is not used for batch configs.")
            batch_config = self.batch
            assert batch_config is not None
            for image_id, template_id in batch_config.image_to_template.items():
                self._validate_image_reference(image_id, f"batch.image_to_template.{image_id}")
                self._validate_image_reference(
                    template_id,
                    f"batch.image_to_template.{image_id}",
                )
                if image_id == template_id:
                    raise ValueError(
                        f"batch.image_to_template.{image_id} must not map an image to itself."
                    )
            return self

        sweep_config = self.sweep
        assert sweep_config is not None
        if self.run.output_root is None:
            raise ValueError("[run].output_root is required for sweep configs.")
        if self.run.output_dir is not None:
            raise ValueError("[run].output_dir is not used for sweep configs.")
        if self.run.output_subdir is not None:
            raise ValueError("[run].output_subdir is not used for sweep configs.")
        self._validate_image_reference(sweep_config.shared_image, "sweep.shared_image")
        if len(set(sweep_config.run_images)) != len(sweep_config.run_images):
            raise ValueError("sweep.run_images entries must be unique.")
        for image_id in sweep_config.run_images:
            self._validate_image_reference(image_id, f"sweep.run_images[{image_id}]")
        if sweep_config.shared_image in sweep_config.run_images:
            raise ValueError("sweep.shared_image must not also appear in sweep.run_images.")
        return self

    def mode_name(self) -> RegistrationMode:
        if self.single is not None:
            return "single"
        if self.batch is not None:
            return "batch"
        return "sweep"

    def _validate_image_reference(self, image_id: str, label: str) -> None:
        if image_id not in self.images:
            raise ValueError(f"{label} references unknown image id: {image_id}")
