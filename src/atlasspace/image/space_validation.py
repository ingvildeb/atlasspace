from __future__ import annotations

from atlasspace.config.space_models import SpaceDefinition


def is_isotropic(space: SpaceDefinition) -> bool:
    x, y, z = space.resolution_um
    return x == y == z


def validate_isotropic_space(space: SpaceDefinition) -> None:
    if not is_isotropic(space):
        raise ValueError(
            "Registration currently supports only isotropic image spaces. "
            f"Got resolution_um={space.resolution_um} for space "
            f"{space.space_name or '<unnamed>'}."
        )
