# Atlasbuilder Repo Plan

## Purpose

`atlasbuilder` is meant to become a reusable package for brain registration,
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
- Use typed config models so lab users can run workflows through readable YAML.
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

Importable package code lives under `src/atlasbuilder/`.

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
  Typed config models, space-definition models, and YAML loading helpers.

### User-facing config assets

Config templates and shipped presets live under `configs/`.

- `configs/registration_presets/`
  Reusable registration method presets.

- `configs/run_presets/`
  Templates for batch and sweep runs.

### Workflow scripts

Opinionated runnable scripts live under `workflows/`. These should stay thin
and mostly orchestrate config loading plus calls into the library package.

### Documentation

Planning notes, design decisions, and user-facing guidance live under `docs/`.


## Config Philosophy

The config system is intentionally split into separate concerns.

- A registration preset defines one complete registration strategy.
- A batch run selects one preset and applies it across many run images using
  one shared image with an explicit fixed/moving role plus run-level
  orientation-alignment policy.
- A sweep run selects many presets and compares them across one or more run
  images using one shared image with an explicit fixed/moving role plus
  run-level orientation-alignment policy.
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

### Batch run

A batch run is represented by `RegistrationBatchConfig` and contains:

- `shared_image_role`
- `orientation_alignment`
- `shared_image`
- `registration_preset`
- `output_subdir_name`
- `run_images`

Batch behavior:

- if `output_subdir_name` is set, output goes to
  `{run_image_parent}/{output_subdir_name}`

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

### Sweep run

A sweep run is represented by `RegistrationSweepConfig` and contains:

- `shared_image_role`
- `orientation_alignment`
- `shared_image`
- `registration_presets`
- `output_root`
- `run_images`

Sweep behavior:

- outputs should be created under deterministic subfolders such as
  `{output_root}/{image_id}_{preset_name}`


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
