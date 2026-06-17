from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import ants
except ImportError as exc:
    raise ImportError(
        "atlasbuilder.transforms.antspy_transformation requires antspyx. "
        "Install atlasbuilder in an environment with ANTsPy available to apply transforms."
    ) from exc


def normalize_transform_paths(transform_paths: list[Path]) -> list[str]:
    return [str(path) for path in transform_paths]


def apply_transform_paths_to_image(
    moving_image: ants.ANTsImage,
    reference_image: ants.ANTsImage,
    transform_paths: list[Path],
    *,
    interpolation: str,
) -> ants.ANTsImage:
    return ants.apply_transforms(
        fixed=reference_image,
        moving=moving_image,
        transformlist=normalize_transform_paths(transform_paths),
        interpolator=interpolation,
    )


def apply_transform_paths_to_points(
    points_xyz: np.ndarray,
    transform_paths: list[Path],
) -> np.ndarray:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (n_points, 3).")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "atlasbuilder.transforms.antspy_transformation requires pandas to transform points."
        ) from exc

    points_df = pd.DataFrame(points_xyz, columns=["x", "y", "z"])
    transformed_df = ants.apply_transforms_to_points(
        dim=3,
        points=points_df,
        transformlist=normalize_transform_paths(transform_paths),
    )
    return transformed_df.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float64, copy=True)
