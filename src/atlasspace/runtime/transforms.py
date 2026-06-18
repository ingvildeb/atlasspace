from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from atlasspace.config.space_models import SpaceDefinition
from atlasspace.runtime.registration import RegistrationJob, RegistrationResult


TransformDirection = Literal["forward", "inverse"]


@dataclass
class TransformSequence:
    source_space: SpaceDefinition
    target_space: SpaceDefinition
    forward_paths: list[Path]
    inverse_paths: list[Path]
    source_transform_space: SpaceDefinition | None = None
    target_transform_space: SpaceDefinition | None = None

    def __post_init__(self) -> None:
        if not self.forward_paths:
            raise ValueError("forward_paths must not be empty.")
        if not self.inverse_paths:
            raise ValueError("inverse_paths must not be empty.")
        if self.source_transform_space is None:
            self.source_transform_space = self.source_space
        if self.target_transform_space is None:
            self.target_transform_space = self.target_space

    @classmethod
    def from_antspy_output(
        cls,
        output_dir: Path,
        *,
        source_space: SpaceDefinition,
        target_space: SpaceDefinition,
        source_transform_space: SpaceDefinition | None = None,
        target_transform_space: SpaceDefinition | None = None,
        prefix: str = "ANTsPy_",
    ) -> "TransformSequence":
        affine_path = output_dir / f"{prefix}0GenericAffine.mat"
        warp_path = output_dir / f"{prefix}1Warp.nii.gz"
        inverse_warp_path = output_dir / f"{prefix}1InverseWarp.nii.gz"

        if not affine_path.exists():
            raise FileNotFoundError(
                "Could not find the ANTs affine transform at "
                f"{affine_path}. Expected an ANTs output folder."
            )

        forward_paths = [affine_path]
        if warp_path.exists():
            forward_paths = [warp_path, affine_path]

        inverse_paths = [affine_path]
        if inverse_warp_path.exists():
            inverse_paths = [affine_path, inverse_warp_path]

        return cls(
            source_space=source_space,
            target_space=target_space,
            forward_paths=forward_paths,
            inverse_paths=inverse_paths,
            source_transform_space=source_transform_space,
            target_transform_space=target_transform_space,
        )

    @classmethod
    def from_registration_result(
        cls,
        result: RegistrationResult,
    ) -> "TransformSequence":
        if not result.success:
            raise ValueError("Cannot build a TransformSequence from an unsuccessful registration result.")
        if not result.forward_transforms:
            raise ValueError("Registration result does not contain forward transforms.")
        if not result.inverse_transforms:
            raise ValueError("Registration result does not contain inverse transforms.")

        return cls(
            source_space=result.declared_moving_space,
            target_space=result.declared_fixed_space,
            forward_paths=list(result.forward_transforms),
            inverse_paths=list(result.inverse_transforms),
            source_transform_space=result.effective_moving_space,
            target_transform_space=result.effective_fixed_space,
        )

    @classmethod
    def from_registration(
        cls,
        job: RegistrationJob,
        result: RegistrationResult,
    ) -> "TransformSequence":
        return cls.from_registration_result(result)

    def inverted(self) -> "TransformSequence":
        return TransformSequence(
            source_space=self.target_space,
            target_space=self.source_space,
            forward_paths=list(self.inverse_paths),
            inverse_paths=list(self.forward_paths),
            source_transform_space=self.target_transform_space,
            target_transform_space=self.source_transform_space,
        )

    def paths_for_direction(self, direction: TransformDirection) -> list[Path]:
        if direction == "forward":
            return list(self.forward_paths)
        if direction == "inverse":
            return list(self.inverse_paths)
        raise ValueError(f"Unsupported transform direction: {direction}")
