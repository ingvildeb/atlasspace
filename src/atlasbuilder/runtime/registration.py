from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from atlasbuilder.config.config_models import (
    ImageConfig,
    OrientationAlignmentMode,
    RegistrationParametersConfig,
)
from atlasbuilder.config.space_models import SpaceDefinition


@dataclass
class RegistrationJob:
    fixed_image_config: ImageConfig
    moving_image_config: ImageConfig
    output_dir: Path
    parameters: RegistrationParametersConfig
    orientation_alignment: OrientationAlignmentMode = "none"


@dataclass
class RegistrationResult:
    fixed_image_id: str
    moving_image_id: str
    preset_name: str
    output_dir: Path
    success: bool
    declared_fixed_space: SpaceDefinition
    declared_moving_space: SpaceDefinition
    effective_fixed_space: SpaceDefinition
    effective_moving_space: SpaceDefinition
    runtime_seconds: float | None = None
    warped_image: Path | None = None
    inverse_warped_image: Path | None = None
    forward_transforms: list[Path] = field(default_factory=list)
    inverse_transforms: list[Path] = field(default_factory=list)
    error_message: str | None = None
