from __future__ import annotations

from pathlib import Path

try:
    import nrrd
except ImportError as exc:
    raise ImportError(
        "atlasbuilder.io.nrrd requires pynrrd. "
        "Install atlasbuilder with pynrrd available to ingest NRRD files."
    ) from exc

from atlasbuilder.config.config_models import ImageConfig
from atlasbuilder.config.space_models import SpaceDefinition
from atlasbuilder.image._image_config_utils import (
    build_output_image_config,
    validate_or_fill_space_shape,
)
from atlasbuilder.io.nifti import write_nifti_from_array


def ingest_nrrd_as_nifti(
    input_path: Path,
    output_path: Path,
    *,
    image_id: str,
    space: SpaceDefinition,
) -> ImageConfig:
    array, _header = nrrd.read(str(input_path))

    provisional_config = ImageConfig(
        image_id=image_id,
        image=output_path,
        space=space,
    )
    output_space = validate_or_fill_space_shape(
        provisional_config,
        tuple(int(v) for v in array.shape),
    )
    write_nifti_from_array(array, output_space, output_path, dtype=array.dtype)
    return build_output_image_config(provisional_config, output_path, output_space)
