# Registration Config Summary

This document summarizes the current registration config system in
`atlasspace`.

The important high-level idea is that registration configuration is split into
two layers:

1. registration presets
2. registration job specs

Presets describe how registration should be performed. Job specs describe what
images should be registered, in what combinations, and where outputs should go.


## 1. Two Config Layers

### Registration presets

Registration presets are YAML files that define one reusable registration
strategy.

They are represented internally by `RegistrationParametersConfig` in:

- `src/atlasspace/config/preset_models.py`

Built-in presets currently live in:

- `src/atlasspace/presets/registration/`

Examples:

- `baseline_syn_kimlab`
- `tuned_syn_cc`

A preset includes three sections:

- `preprocessing`
- `registration`
- `execution`

In practice, a preset answers:

- how should the images be preprocessed?
- what affine and deformable settings should be used?
- what runtime settings should be used?


### Registration job specs

Registration job specs are TOML files that define a concrete run.

They are represented internally by `RegistrationJobSpecConfig` in:

- `src/atlasspace/config/job_spec_models.py`

Example job-spec templates live in:

- `examples/configs/registration_single_template.toml`
- `examples/configs/registration_batch_template.toml`
- `examples/configs/registration_sweep_template.toml`

A job spec answers:

- which images are involved?
- which preset or presets should be used?
- which image is fixed and which is moving?
- what output directory should be used?
- is this a single run, a batch run, or a sweep?


## 2. Normalization Flow

The intended flow is:

1. Load a TOML job spec.
2. Validate the TOML structure.
3. Resolve inherited image defaults.
4. Resolve fixed/moving image pairs.
5. Normalize into a `RegistrationPlan`.
6. Expand the plan into concrete `RegistrationJob` objects.

This lets `single`, `batch`, and `sweep` share one execution path after
normalization.


## 3. Canonical Job-Spec Structure

Every registration TOML job spec uses the same top-level structure:

- `[run]`
- optional `[image_defaults]`
- optional `[moving_segmentations]`
- `[images.<image_id>]`
- exactly one of:
  - `[single]`
  - `[batch]`
  - `[sweep]`


## 4. Common Sections

### `[run]`

This section contains shared run-level settings.

Current fields:

- `registration_presets`
  - list of preset names or preset file paths
- `orientation_alignment`
  - one of:
    - `"none"`
    - `"moving_to_fixed"`
    - `"fixed_to_moving"`
- `write_input_images`
  - boolean
- `output_dir`
  - used by `single`
- `output_root`
  - used by `batch` and `sweep`

Notes:

- `single` and `batch` currently require exactly one preset.
- `sweep` supports multiple presets.


### `[image_defaults]`

This section provides shared defaults for images that do not explicitly define
their own values.

Current fields:

- `orientation`
- `resolution_um`

This is mainly a convenience section so users do not need to repeat the same
resolution or orientation on every image when those values are shared.


### `[moving_segmentations]`

This section stores policy for propagating segmentations attached to the moving
image.

Current fields:

- `enabled`
- `interpolation`
  - currently:
    - `"genericLabel"`
    - `"nearestNeighbor"`
- `output_subdir`
- `write_intermediates`

Important note:

- this section is already part of the canonical config and is preserved in the
  normalized plan
- automatic post-registration propagation of these segmentations is not yet
  fully wired into the default registration execution path


### `[images.<image_id>]`

Each image lives under the shared `[images]` namespace.

Current fields:

- `image`
- optional `space_name`
- optional `orientation`
- optional `resolution_um`

Images may also define attached segmentations:

```toml
[images.template_p56.segmentations]
brain_mask = "/path/to/T_P56_mask.nii.gz"
annotation = "/path/to/T_P56_annotation.nii.gz"
```

These segmentations inherit the same declared image space as their parent
image.


## 5. The Three Modes

### `single`

`single` is the simplest mode.

It explicitly declares one fixed image and one moving image:

```toml
[single]
fixed_image = "neun_p14"
moving_image = "neun_p56"
```

Use `single` when you want one registration with one preset.


### `batch`

`batch` is for many registrations using one preset.

It uses:

- `template_role`
- `image_to_template`

Example:

```toml
[batch]
template_role = "moving"
image_to_template = { subject_a = "template_p56", subject_b = "template_p56" }
```

Interpretation:

- keys are run-image ids
- values are template-image ids
- `template_role` decides whether the template is treated as fixed or moving

Use `batch` when you want one preset applied across many image-template pairs.


### `sweep`

`sweep` is for comparing multiple presets on one shared image plus one or more
run images.

It uses:

- `shared_image`
- `shared_image_role`
- `run_images`

Example:

```toml
[sweep]
shared_image = "neun_p14"
shared_image_role = "fixed"
run_images = ["neun_p56"]
```

Interpretation:

- the shared image is reused for every run image
- every preset listed in `[run].registration_presets` is applied to every pair

Use `sweep` when you want to compare registration strategies rather than only
execute one strategy.


## 6. How Fixed And Moving Are Resolved

The fixed/moving rule depends on the mode.

### `single`

- fixed = `[single].fixed_image`
- moving = `[single].moving_image`

### `batch`

For each mapping `run_image -> template_image`:

- if `template_role = "moving"`:
  - fixed = run image
  - moving = template image
- if `template_role = "fixed"`:
  - fixed = template image
  - moving = run image

### `sweep`

For each `run_image`:

- if `shared_image_role = "fixed"`:
  - fixed = shared image
  - moving = run image
- if `shared_image_role = "moving"`:
  - fixed = run image
  - moving = shared image


## 7. Runtime Models

After validation and normalization, the user-facing TOML is converted into a
small number of runtime-facing models.

### `ImageConfig`

Defined in:

- `src/atlasspace/config/image_models.py`

This is the resolved runtime image model.

It contains:

- `image_id`
- `image`
- `space`
- optional `segmentations`


### `RegistrationPlan`

Defined in:

- `src/atlasspace/config/job_spec_models.py`

This is the normalized representation of a run.

It contains:

- `mode`
- `preset_references`
- `orientation_alignment`
- `write_input_images`
- `single_output_dir` or `output_root`
- resolved `images`
- resolved `pairs`
- `moving_segmentations`

This is the model that bridges user config and execution.


### `RegistrationJob`

Defined in:

- `src/atlasspace/runtime/registration.py`

This is the concrete executable unit used by the runner.

Each `RegistrationJob` contains:

- one fixed image
- one moving image
- one loaded registration preset
- one concrete output directory


## 8. Output Layout

Output layout is mode-dependent.

### `single`

`single` writes to:

- `[run].output_dir`

### `batch` and `sweep`

`batch` and `sweep` write to:

- `{output_root}/{fixed_image_id}__{moving_image_id}/{preset_name}`

This keeps output naming consistent across the normalized execution path.


## 9. Current File Responsibilities

The config-related modules are currently split like this:

- `config/preset_models.py`
  - registration preset models
- `config/image_models.py`
  - resolved runtime image model plus shared fixed/moving role literals
- `config/job_spec_models.py`
  - TOML job-spec models plus normalized plan models
- `config/space_models.py`
  - declared space metadata
- `config/config_loading.py`
  - preset loading, TOML loading, and job-spec normalization


## 10. Practical Guidance

When deciding what to edit:

- edit a preset YAML if you want to change registration behavior
- edit a TOML job spec if you want to change which images are run, how they
  are paired, or where outputs go

In other words:

- presets answer "how"
- job specs answer "what"


## 11. Current Examples

The current canonical example configs are:

- `examples/configs/registration_single_template.toml`
- `examples/configs/registration_batch_template.toml`
- `examples/configs/registration_sweep_template.toml`

The corresponding example scripts are:

- `examples/registration_single_run_example.py`
- `examples/registration_batch_run_example.py`
- `examples/registration_sweep_run_example.py`


## 12. Likely Future Updates

This document reflects the current cleaned-up config architecture.

Likely future refinements include:

- wiring moving-segmentation propagation fully into the default execution path
- adding more user-facing docs around transform application after registration
- updating sibling repos such as HPC wrappers to consume the canonical
  `RegistrationPlan` path directly
