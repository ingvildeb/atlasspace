# atlasspace

`atlasspace` is a reusable Python package for spatial image preparation, registration, transform application, and template-space workflows for volumetric brain data.

`atlasspace` currently supports:

- explicit spatial metadata handling through `SpaceDefinition` and `ImageConfig`
- reusable image operations such as masking, reorientation, resampling, resizing, and symmetry helpers
- ANTsPy-based registration helpers
- ANTsPy-based transform application for images, segmentations, and point coordinates
- iterative template refinement using confidence-based weighted averaging

## Core concepts

### Spatial metadata

`SpaceDefinition` stores the spatial metadata of an image volume. This includes:

- orientation (following [BrainGlobe three-letter convention](https://brainglobe.info/documentation/setting-up/image-definition.html#orientation))
- resolution in microns
- optional `space_name`
- optional `shape`

### Image configuration

`ImageConfig` stores metadata about an image through:
- a `path`
- an `image_id`
- a `SpaceDefinition`

Most public image, registration, transform, and template helpers operate on `ImageConfig`.

## Installation

`atlasspace` is written for Python 3.10+.

To get started, create a dedicated conda environment:

```bash
conda create -n atlasspace python=3.11
conda activate atlasspace
```

For a local editable install during development or beta testing, use:

```bash
pip install -e .
```

The intended packaged usage pattern will be:

```bash
pip install atlasspace
```

This installs the core package, including the ANTsPy-based registration and transform functionality:

```bash
pip install -e .
```

## Example registration workflow

A minimal registration workflow looks like this:

```python
from pathlib import Path

from atlasspace import ImageConfig, SpaceDefinition, registration

preset = registration.load_preset("tuned_syn_cc")

fixed_image = ImageConfig(
    image_id="subject_001",
    image=Path("subject_001.nii.gz"),
    space=SpaceDefinition(
        space_name="subject_001",
        orientation="las",
        resolution_um=(20.0, 20.0, 20.0),
    ),
)

moving_image = ImageConfig(
    image_id="template_p56",
    image=Path("template_p56.nii.gz"),
    space=SpaceDefinition(
        space_name="template_p56",
        orientation="lsp",
        resolution_um=(20.0, 20.0, 20.0),
    ),
)

job = registration.RegistrationJob(
    fixed_image_config=fixed_image,
    moving_image_config=moving_image,
    output_dir=Path("outputs/subject_001_registration"),
    parameters=preset,
    orientation_alignment="fixed_to_moving",
)

result = registration.run_antspy_registration(job)

print(result.success)
print(result.warped_image)
print(result.inverse_warped_image)
```

## Documentation

More detailed documentation is in the works and will be found under `docs/`.

## License

This project is released under the terms of the [LICENSE](LICENSE).
