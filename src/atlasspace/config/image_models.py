from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from atlasspace.config.space_models import SpaceDefinition


ImageRole = Literal["fixed", "moving"]
OrientationAlignmentMode = Literal["none", "moving_to_fixed", "fixed_to_moving"]


class ImageConfig(BaseModel):
    image_id: str
    image: Path
    space: SpaceDefinition
    segmentations: dict[str, Path] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_image(self) -> "ImageConfig":
        if not self.image_id.strip():
            raise ValueError("image_id must not be empty.")
        if any(not segmentation_id.strip() for segmentation_id in self.segmentations):
            raise ValueError("segmentation ids must not be empty.")
        return self
