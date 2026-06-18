from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


AxisLabel = Literal["x", "y", "z"]


class SpaceDefinition(BaseModel):
    space_name: str | None = None
    orientation: str
    axis_labels: tuple[AxisLabel, AxisLabel, AxisLabel] = ("x", "y", "z")
    units: str = "voxel"
    resolution_um: tuple[float, float, float]
    shape: tuple[int, int, int] | None = None

    @model_validator(mode="after")
    def validate_space(self) -> "SpaceDefinition":
        if self.space_name is not None and not self.space_name.strip():
            raise ValueError("space_name must not be empty when provided.")

        orientation = self.orientation.strip().lower()
        if len(orientation) != 3:
            raise ValueError("orientation must have length 3.")

        valid_letters = {"l", "r", "a", "p", "s", "i"}
        if any(letter not in valid_letters for letter in orientation):
            raise ValueError(
                "orientation must use BrainGlobe-style letters from {l, r, a, p, s, i}."
            )

        orientation_families = {
            ("l" if letter in {"l", "r"} else "a" if letter in {"a", "p"} else "s")
            for letter in orientation
        }
        if orientation_families != {"l", "a", "s"}:
            raise ValueError(
                "orientation must include one left/right, one anterior/posterior, "
                "and one superior/inferior axis."
            )

        axis_set = set(self.axis_labels)
        if axis_set != {"x", "y", "z"}:
            raise ValueError("axis_labels must contain x, y, and z exactly once.")

        if any(value <= 0 for value in self.resolution_um):
            raise ValueError("resolution_um values must all be positive.")

        if self.shape is not None and any(value <= 0 for value in self.shape):
            raise ValueError("shape values must all be positive when provided.")

        self.orientation = orientation
        return self
