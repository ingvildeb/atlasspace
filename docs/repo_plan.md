# atlasspace Repo Plan

## Purpose

`atlasspace` is meant to become a reusable package for brain registration,
template generation, and related atlas-space workflows. The guiding goal is to
separate reusable computational building blocks from pipeline-specific scripts,
so the same package can support simple one-off use cases, troubleshooting, and
larger repeatable workflows.

The long-term scope can grow beyond registration and template generation into
areas like segmentation and atlas-space analysis, but the initial focus is on:

- reusable registration helpers
- reusable template-building helpers
- clean config-driven workflows on top of the reusable core


## Design Principles

- Keep reusable library code separate from workflow-specific scripts.
- Prefer small, composable functions over monolithic pipeline scripts.
- Use typed config models so lab users can run workflows through readable TOML
  job specs plus reusable YAML registration presets.
- Keep the package usable both from config files and directly from Python.
- Build module by module instead of creating large placeholder code upfront.


## Naming Conventions

- Use `*_models.py` for files that primarily define structured model classes,
  such as dataclasses or Pydantic models.
- Use process-oriented noun names for reusable function modules, such as
  `config_loading.py` or `preprocessing.py`.
- Reserve explicit verb phrases primarily for runnable scripts and entrypoints,
  especially under `workflows/`.


## High-Level Layout

### Library code

Importable package code lives under `src/atlasspace/`.

- `registration/`
  Registration execution helpers, batching, parameter sweeps, transform
  application, and result summaries.

- `image/`
  Shared image operations such as intensity handling, masks, resampling, and
  symmetry helpers.

- `template/`
  Template averaging, confidence mapping, blending, and support-map logic.

- `io/`
  NIfTI helpers, manifest helpers, and path-related utilities.

- `config/`
  Typed preset models, image models, job-spec models, space-definition models,
  and config normalization helpers.

### User-facing config assets

Built-in registration presets and canonical TOML job-spec templates are both
shipped inside the installable package.

- `src/atlasspace/presets/registration/`
  Built-in reusable registration method presets shipped with the package.

- `src/atlasspace/config_templates/registration_single_template.toml`
  Canonical starter template for one registration run using one preset.

- `src/atlasspace/config_templates/registration_batch_template.toml`
  Canonical starter template for batch registration runs.

- `src/atlasspace/config_templates/registration_sweep_template.toml`
  Canonical starter template for registration sweep runs.

### Workflow scripts

Opinionated runnable scripts live under `workflows/`. These should stay thin
and mostly orchestrate config loading plus calls into the library package.

### Documentation

Planning notes, design decisions, and user-facing guidance live under `docs/`.


## Config Philosophy

The config system is intentionally split into separate concerns.

- A registration preset defines one complete registration strategy.
- A registration job spec is written in TOML and normalizes into a
  `RegistrationPlan`.
- A `single` run selects one preset and one fixed/moving image pair.
- A `batch` run selects one preset and applies it across many run-image to
  template-image mappings.
- A `sweep` run selects many presets and compares them across one shared image
  plus one or more run images.
- Image-space metadata is represented explicitly through a reusable
  `SpaceDefinition` model rather than loose resolution/orientation fields.

This avoids ambiguous designs where a run config both selects a preset and
overrides fields inside it.


## Current Agreed Config Shape

### Registration preset

A registration preset is represented by `RegistrationParametersConfig` and
contains:

- `name`
- `description`
- `preprocessing`
- `registration`
- `execution`

### Image metadata

An image entry is represented by `ImageConfig` and contains:

- `image_id`
- `image`
- `space`

where `space` is represented by `SpaceDefinition` and contains:

- optional `space_name`
- `orientation`
- `axis_labels`
- `units`
- `resolution_um`
- optional `shape`

Current registration assumption:

- registration currently validates isotropic image spacing and uses an explicit
  orientation-alignment policy (`none`, `moving_to_fixed`, or
  `fixed_to_moving`) before intensity preprocessing
- the runner always writes normalized registration-input NIfTIs that reflect
  the declared `SpaceDefinition` and any requested reorientation

### Canonical job spec

The canonical TOML document is represented by `RegistrationJobSpecConfig` and
contains:

- `[run]`
- optional `[image_defaults]`
- optional `[moving_segmentations]`
- `[images.<image_id>]`
- exactly one of `[single]`, `[batch]`, or `[sweep]`

This normalizes into `RegistrationPlan`, which contains:

- `mode`
- `preset_references`
- `orientation_alignment`
- `write_input_images`
- mode-specific output placement fields
- resolved `images`
- resolved `pairs`
- `moving_segmentations`

Normalized output behavior:

- `single` uses `single_output_dir`
- `batch` uses either `{run_image.parent}/{output_subdir}` or
  `{output_root}/{fixed_image_id}__{moving_image_id}/{preset_name}`
- `sweep` uses `{output_root}/{fixed_image_id}__{moving_image_id}/{preset_name}`


## Current Registration Presets

The first two shipped presets are:

- `baseline_syn_kimlab`
  Historical Kim lab SyN baseline benchmark.

- `tuned_syn_cc`
  Conservative CC-based tuned preset selected from troubleshooting work.


## Planned Implementation Sequence

The package is being built deliberately in stages.

1. Scaffold repo layout
2. Define config models and preset templates
3. Add config loading helpers
4. Implement registration module
5. Implement shared image helpers as needed by registration/template code
6. Implement template module
7. Add thin workflow scripts on top


## Near-Term Next Steps

- add package `__init__` files only when they become useful
- evaluate the config-driven batch example against the working ANTsPy runner
- decide which transform-related helper(s) should be extracted from
  `antspy_registration.py` next
- keep validating design decisions in docs as the package grows
