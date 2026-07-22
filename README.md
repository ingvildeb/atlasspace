# atlasspace

`atlasspace` is a reusable Python package for image preparation, registration, transform application, and template-space workflows for volumetric brain data.

This is an early-stage, pre-1.0 release. The core architecture is in place, but
some APIs and workflow patterns may still evolve as the package is tested in
real use.

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

- an image path (`image`)
- an `image_id`
- a `SpaceDefinition`

Most public image, registration, transform, and template helpers operate on `ImageConfig`.

For example, first define the space occupied by a 20-micron subject image:

```python
from pathlib import Path

from atlasspace import ImageConfig, SpaceDefinition

subject_space = SpaceDefinition(
    space_name="subject_001",
    orientation="las",
    resolution_um=(20.0, 20.0, 20.0),
    shape=(640, 400, 580),
)
```

Then use that space definition when configuring the image:

```python
subject_image = ImageConfig(
    image_id="subject_001_autofluorescence",
    image=Path("data/subject_001_autofluorescence.nii.gz"),
    space=subject_space,
)
```

The resulting `subject_image` keeps the image path and identity together with
the explicit spatial metadata that downstream operations need.

## Example image operations

Image operations accept an `ImageConfig`, write the processed image to the
requested path, and return a new `ImageConfig` describing that output.

### Reorient an image

To reorient `subject_image` from `las` orientation to match a template in
`lsp` orientation, define the target space and pass it to
`reorient_image_to_match`:

```python
from atlasspace import SpaceDefinition
from atlasspace.image import reorient_image_to_match

template_space = SpaceDefinition(
    space_name="template_p56",
    orientation="lsp",
    resolution_um=(20.0, 20.0, 20.0),
)

reoriented_image = reorient_image_to_match(
    subject_image,
    target_space=template_space,
    output_path=Path("outputs/subject_001_lsp.nii.gz"),
)
```

The returned config points to the new file and contains the reoriented spatial
metadata, including any corresponding changes to axis order and shape.

### Resample an image

To resample the reoriented image from 20-micron to 50-micron isotropic
resolution:

```python
from atlasspace.image import resample_image_to_resolution

resampled_image = resample_image_to_resolution(
    reoriented_image,
    output_path=Path("outputs/subject_001_lsp_50um.nii.gz"),
    target_resolution_um=(50.0, 50.0, 50.0),
    interpolation="linear",
)
```

`resampled_image.space` contains the new resolution and shape. For label or
segmentation images, use `interpolation="nearest"` to preserve discrete label
values.

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

## Apply an existing registration

An existing ANTsPy output directory can be loaded as a `TransformSequence` and
reused to transform another image. Continuing from the registration example
above, this applies the saved template-to-subject transforms to a second image
in the moving template space:

```python
from atlasspace.transforms import TransformSequence, transform_image

registration_dir = Path("outputs/subject_001_registration")

transforms = TransformSequence.from_registration_output(registration_dir)

template_channel_2 = ImageConfig(
    image_id="template_p56_channel_2",
    image=Path("template_p56_channel_2.nii.gz"),
    space=moving_image.space,
)

warped_channel_2 = transform_image(
    template_channel_2,
    transforms,
    reference_config=fixed_image,
    direction="forward",
    interpolation="linear",
    output_path=registration_dir / "template_p56_channel_2_Warped.nii.gz",
)
```

Here, `forward` maps from the registration's moving/source space into its
fixed/target space. `from_registration_output` loads the declared and effective
spaces, along with the ordered transform paths, from the
`registration_result.json` manifest written by atlasspace.

Legacy atlasspace outputs that contain `registration_summary.txt` can be
upgraded once without changing their registration products:

```python
from atlasspace import registration

registration.migrate_legacy_registration_output(registration_dir)
```

The migration records its legacy metadata assumptions in the new manifest and
does not overwrite an existing manifest by default.

## Documentation

More detailed documentation is in the works and will be found under `docs/`.

## License

This project is released under the terms of the [LICENSE](LICENSE).
