# Registration Module Plan

## Goal

The registration module should make it easy for lab users to do two main kinds
of work:

1. run one registration method on many images
2. run multiple registration methods on one or a few images for comparison

This module is therefore designed around reusable registration presets plus a
canonical TOML job-spec layer that supports `single`, `batch`, and `sweep`
workflows.


## Scope For V1

The initial registration module should focus on:

- config-driven execution
- preprocessing helpers
- ANTsPy-based registration helpers
- single, batch, and sweep execution through one normalized plan layer
- result summarization

It should not yet include the more complex initialization/chained-transform
logic from the troubleshooting repo, because that was not clearly beneficial in
practice and would add substantial complexity.


## User-Facing Concepts

### Registration preset

A preset is a complete registration strategy stored as YAML and loaded through
`RegistrationParametersConfig` from `config/preset_models.py`.

Examples:

- `baseline_syn_kimlab`
- `tuned_syn_cc`

Built-in presets are now shipped inside the package under
`src/atlasspace/presets/registration/`, while users can still point to their
own custom preset YAML files by path.

### Registration job spec

Run configuration is expressed through TOML and normalized before jobs are
built.

Supported modes:

- `single`
- `batch`
- `sweep`


## Config Models

The config model definitions live in:

- `src/atlasspace/config/preset_models.py`
- `src/atlasspace/config/image_models.py`
- `src/atlasspace/config/job_spec_models.py`
- `src/atlasspace/config/space_models.py`

The runtime execution models planned for the registration module should live in:

- `src/atlasspace/runtime/registration.py`

### `RegistrationParametersConfig`

Represents one full registration method definition.

Sections:

- `preprocessing`
- `registration`
- `execution`

### `ImageConfig`

Represents one resolved runtime image entry.

Fields:

- `image_id`
- `image`
- `space`
- optional `segmentations`

### `SpaceDefinition`

Represents explicit image-space metadata shared across registration inputs and
future transform-related logic.

Fields:

- optional `space_name`
- `orientation`
- `axis_labels`
- `units`
- `resolution_um`
- optional `shape`

Notes:

- `orientation` uses the same BrainGlobe-style convention as
  `lsfm_cell_mapping`, where each letter describes the anatomical side at voxel
  index 0 for that axis.
- `resolution_um` is stored explicitly per axis rather than as a single scalar
  so the model can represent anisotropic images cleanly.

### `RegistrationJobSpecConfig`

Represents the user-authored TOML registration document.

It contains:

- `[run]`
- optional `[image_defaults]`
- optional `[moving_segmentations]`
- `[images.<image_id>]`
- exactly one of `[single]`, `[batch]`, or `[sweep]`

### `RegistrationPlan`

Represents the normalized execution plan after defaults, image references, and
fixed/moving pair semantics have been resolved.


## Preprocessing Plan

The preprocessing section should support:

- `intensity_normalization`
  - allowed values for now: `zscore`, `robust_zscore`

- `minmax_clip_percentiles`
  - optional percentile clipping and rescaling

- `histogram_match`
  - optional pre-registration histogram matching

- `gaussian_sigma_vox`
  - optional smoothing before registration

These are treated as separate knobs because they affect image preparation in
different ways and are useful for troubleshooting different failure modes.

Before intensity preprocessing, registration should also perform orientation
alignment:

- the reorientation policy is explicit and lives with the run/job definition
  rather than the reusable registration preset
- supported policies are `none`, `moving_to_fixed`, and `fixed_to_moving`
- orientations use the BrainGlobe origin-based convention
- current registration support validates isotropic image spacing explicitly
- the runner writes normalized registration-input NIfTIs before ANTs consumes
  the images


## Registration Settings Plan

The registration section should support:

- `working_resolution_um`
- `transform_type`
- affine-stage settings
- deformable-stage settings

Supported transform types for v1:

- `Rigid`
- `Affine`
- `SyN`
- `SyNOnly`

Supported metric values for v1:

- affine metric: `mattes`
- deformable metric: `mattes`, `CC`

The package should start with a curated set rather than exposing every ANTs
option immediately.


## Execution Settings Plan

The execution section should support:

- `threads`
- `singleprecision`
- `use_legacy_histogram_matching`
- `verbose`
- `random_seed`
- `write_input_images`

These are kept separate from registration settings because they relate more to
runtime behavior and reproducibility than to the conceptual registration method.

The prepared registration inputs are always saved as:

- `fixed_normalized_for_registration.nii.gz`
- `moving_normalized_for_registration.nii.gz`


## Planned Module Structure

Inside `src/atlasspace/registration/`, the current intended structure is:

- `presets.py`
  Optional helpers for named presets, if needed beyond YAML.

- `job_building.py`
  Expansion of normalized `RegistrationPlan` objects into concrete
  registration jobs.

- `preprocessing.py`
  Normalization, clipping, smoothing, histogram matching, resampling helpers.

- `transforms.py`
  Transform application and transform-list helpers.

- `antspy_registration.py`
  ANTsPy-specific registration execution.

Related runtime dataclasses live in:

- `src/atlasspace/runtime/registration.py`

- `results.py`
  Summaries and conversion of run results into tabular form.


## Relationship To The Troubleshooting Repo

The registration module is informed by the work in
`misc_data_processing/registration_troubleshooting`, but is not intended to
mirror that repo's script structure directly.

The key ideas being carried over are:

- explicit parameter presets
- reusable preprocessing helpers
- shared parameter sets across many runs
- parameter-comparison workflows
- image-to-image registration rather than subject/template-only assumptions

The key complexity being intentionally deferred is:

- initialization strategies that require chained transforms or more complex
  bookkeeping


## Immediate Next Steps

- evaluate the first ANTsPy runner against real example jobs
- decide which transform-related helper(s) should be extracted into a reusable
  transform module next
