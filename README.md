# atlasbuilder

Tools for working with volumetric brain data as part of atlas generation,
registration, and atlas-based mapping workflows.

## What this repo is for

`atlasbuilder` is a reusable Python package for atlas-space image workflows.
The current core focus is:

- explicit image-space metadata through `ImageConfig` and `SpaceDefinition`
- reusable image operations such as masking, reorientation, resampling, and symmetry
- ANTsPy-based registration helpers
- template confidence mapping, weighted averaging, and template blending

The guiding design principle is to keep reusable library code here while
keeping more opinionated project workflows in downstream repos.

## Main concepts

### `SpaceDefinition`

`SpaceDefinition` stores explicit spatial metadata for a volume, including:

- `orientation`
- `resolution_um`
- optional `space_name`
- optional `shape`

This keeps image-space assumptions explicit rather than scattering resolution
and orientation values across unrelated functions.

### `ImageConfig`

`ImageConfig` pairs an image path with an `image_id` and a `SpaceDefinition`.
Most public image, template, and registration helpers operate on
`ImageConfig` rather than loose file paths.

### Config vs runtime models

The repo separates:

- `config/`
  user-authored configuration models and YAML loading helpers
- `runtime/`
  execution-time dataclasses such as registration jobs/results and template
  accumulation results

## Current module layout

Importable package code lives under `src/atlasbuilder/`.

- `config/`
  config models, space-definition models, and YAML loading helpers
- `image/`
  reusable image operations such as masking, reorientation, pose
  standardization, resampling, resizing, and symmetry
- `io/`
  NIfTI and NRRD helpers
- `registration/`
  registration execution helpers and job-building utilities
- `runtime/`
  runtime dataclasses used by registration and template workflows
- `template/`
  confidence mapping, weighted averaging, blending, and related template logic

## Current workflow areas

The repo currently supports two main kinds of reusable workflow building blocks.

### Registration

Registration is config-driven and centered on:

- reusable registration presets in `configs/registration_presets/`
- batch and sweep run configs
- `run_antspy_registration(...)`
- `build_batch_jobs(...)` and `build_sweep_jobs(...)`

Examples live under `examples/`:

- `registration_single_run_example.py`
- `registration_batch_run_example.py`
- `registration_sweep_run_example.py`

### Template updating

The template module currently focuses on:

- voxelwise confidence-map generation from subject-template agreement
- conversion of confidence maps to weight maps
- weighted and plain template averaging
- blending of new-brain averages with an existing template
- symmetry helpers used after blending

The confidence-map method currently works by:

1. histogram-matching the subject to the template within a valid mask
2. normalizing subject and template to a shared `0-1` scale
3. smoothing both normalized volumes
4. computing a relative residual
5. remapping residuals to confidence inside the valid mask

The current default residual mode is:

`abs(subject - template) / (template + 0.10)`

where `subject` and `template` are the smoothed, normalized arrays.

## Examples and supporting scripts

- `examples/`
  runnable examples and example config files for registration workflows
- `workflows/`
  thin helper scripts that demonstrate specific image-preparation workflows

Downstream repos can build more specialized workflows on top of these library
functions without needing to reimplement the core spatial logic.

## Current status

The repo is now beyond the initial planning phase and has working code for:

- registration
- image preparation and space handling
- template confidence-map generation
- weighted template updating

Documentation is still evolving alongside the code, so the examples and the
`docs/` notes are both useful references.
