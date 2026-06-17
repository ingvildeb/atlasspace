# Atlasbuilder Project Status

## Current identity

Atlasbuilder is now functioning as a reusable spatial-image infrastructure package rather than only a template-building repo. The package currently covers:

- explicit spatial metadata through `ImageConfig` and `SpaceDefinition`
- ANTsPy-based registration
- registration-oriented preprocessing
- reusable image operations such as masking, mirroring, reorientation, pose standardization, resampling, and resizing
- template confidence mapping and weighted template updating
- NIfTI/NRRD I/O helpers

The intended ecosystem role is:

- `atlasbuilder`: registration, transforms, spatial image preparation, and template operations
- `atlaslevels`: atlas label / hierarchy semantics
- `lsfm_cell_mapping`: downstream representation and quantification that can consume atlasbuilder

We discussed a possible future rename to `atlasspace`, but that has been intentionally deferred.

## Current structure

The package has largely settled into the following module groups:

- `config/`: user-facing configuration schemas
- `runtime/`: jobs, results, and accumulation dataclasses
- `image/`: reusable image-space operations
- `registration/`: job building and the ANTsPy runner
- `template/`: confidence maps and weighted averaging/blending
- `io/`: NIfTI and NRRD helpers

This is now a fairly coherent split between configuration, runtime records, reusable image utilities, and workflow-specific logic.

## Important design decisions

### 1. Image- and space-centric modeling

We generalized earlier registration-specific naming toward `ImageConfig`, with `SpaceDefinition` carrying image metadata such as:

- `space_name`
- orientation
- resolution
- optional shape

This was done so atlasbuilder can support not only subject-to-template registration, but also template-to-template and other spatial workflows.

### 2. Explicit metadata over implicit headers

For v1, users explicitly declare important spatial metadata rather than relying on NIfTI headers as ground truth. This was especially important for:

- orientation handling
- reorientation before registration
- later transform-aware workflows

Header parsing may be added later, but was intentionally not made the basis of core behavior yet.

### 3. Registration presets vs run-time policy

We separated reusable registration method presets from run-specific image preparation policy.

In particular:

- registration presets hold ANTs method settings
- orientation alignment / reorientation policy belongs at the batch, sweep, or job layer
- `output_dir` belongs to runtime jobs, not image config

This keeps presets reusable across different labs or conventions.

### 4. ANTs-native backend, generic conceptual layer

Atlasbuilder is currently ANTs-native in implementation. That is an intentional and acceptable v1 choice. We did not try to make the implementation backend-agnostic prematurely.

At the same time, we began sketching a more general transform layer so that public concepts can remain broader than the ANTs backend:

- transform sequences
- transform application to images, segmentations, and points
- inversion / concatenation of transform sequences

This transform layer is now implemented in a first ANTsPy-backed form and has
been validated on real registration outputs for forward and inverse image
transforms plus forward and inverse segmentation transforms.

### 5. Reusable image operations stay narrow

We kept reusable `image/` functions focused and composable rather than building large "do everything" helpers. This includes separate functions for:

- binary masking
- segmentation-to-mask conversion
- axis-code reorientation
- pose standardization
- mirror-unilateral-mask operations
- symmetry helpers
- resampling and resizing

Workflow scripts are expected to compose these functions as needed.

## Registration work completed

Atlasbuilder now has a working ANTsPy registration path with:

- config loading for registration presets
- runtime job and result dataclasses
- preparation of canonical normalized input images for registration
- optional writing of run summaries / preset snapshots
- example scripts for single runs, batch runs, and sweep runs

The registration stack was validated against earlier troubleshooting results, including reorientation behavior. Reorientation support is now working in the registration runner after moving the logic toward declared-space-based preparation.

The registration result model now also carries both declared and effective
spaces, which makes it possible to construct transform sequences that mirror
the actual registration geometry when orientation alignment was used.

## Transform work completed

Atlasbuilder now has a reusable transform layer built around `TransformSequence`.

Implemented pieces include:

- `TransformSequence`
- `TransformSequence.from_antspy_output(...)`
- `TransformSequence.from_registration_result(...)`
- `TransformSequence.from_registration(...)`
- `transform_image(...)`
- `transform_segmentation(...)`
- `transform_points(...)`
- `invert_transform_sequence(...)`
- `concatenate_transform_sequences(...)`

The important settled semantic choice is that transform application is
registration-faithful:

- forward application reproduces `Warped`
- inverse application reproduces `InverseWarped`
- if registration used orientation alignment, atlasbuilder reorients inputs
  into the effective registration spaces before applying transforms
- outputs remain in those effective registration spaces by default rather than
  being automatically restored to the originally declared orientations

This keeps the transform layer aligned with what registration actually did and
makes transform outputs directly comparable to the saved registration products.

## Image and preprocessing work completed

The following reusable image functionality has been implemented:

- `apply_binary_mask(...)`
- `segmentation_to_binary_mask(...)`
- axis-code-based reorientation
- pose standardization using landmarks
- `mirror_unilateral_mask(...)`
- symmetry helpers
- NRRD reading helpers
- NIfTI writing helpers
- generic resampling and resizing operations

We also agreed on strict isotropic-resolution validation for registration inputs in v1.

## Template work completed

### Confidence maps

The confidence-map workflow has been substantially refined. The currently preferred method is:

1. histogram-match subject to template inside mask (optional, currently used)
2. normalize subject and template into a shared comparison space using template-derived bounds
3. smooth
4. compute residuals in normalized space
5. convert residuals to confidence
6. convert confidence to weight maps downstream

The current preferred residual logic is the simplified relative-residual formulation with a floor of `0.10`, without the earlier histogram-adjustment penalty.

### Weighted template updating

Template updating has been modularized into:

- confidence map generation
- weighted template building from those confidence maps

We did substantial debugging here and identified several important issues:

- support count must mean actual positive-weight contribution, not simply presence inside the valid mask
- weighted template updating must use normalized subject volumes, not raw registered brains
- the template update math can happen in normalized comparison space, but the final saved updated template should be mapped back into template intensity space
- the final updated template should use the simpler post-blend reflection average rather than the more complicated support-weighted symmetry helper

The active workflow in `lsfm_atlas_framework` was brought back closer to the originally validated method while preserving the modular split and newer cleanup.

## Related workflow integration completed

Atlasbuilder-based scripts were created in `lsfm_atlas_framework` for:

- masking
- mirroring masks
- pose standardization
- preparing CCFv3 for registration
- building confidence maps
- building weighted templates from confidence
- registering subject brains to a template using tuned registration settings

This means atlasbuilder is already functioning as the reusable core for the next phase of template and registration work in that repo.

We also removed the older `SING_scripts/template_generation` workflow after confirming it had been effectively replaced.

## Important deferred topics (pins)

These are the main things we explicitly put a pin in and should revisit later.

### 1. Deeper provenance / output metadata

The transform layer now exists, but we still have a related deferred topic
around making outputs and provenance more explicit.

This includes continuing to improve:

- how source/target/effective spaces are reflected in saved outputs
- how much transform and template-update provenance should be promoted into
  reusable first-class records
- how self-describing saved outputs should become over time
We also discussed keeping atlasbuilder distinct from
`brainglobe-ccf-translator`:

- atlasbuilder should own local registration-derived transform application
- `brainglobe-ccf-translator` should not be duplicated as a global atlas-space graph system

### 2. Naming / repo rename

We brainstormed broader names for the repo and liked `atlasspace` best. However, we explicitly deferred renaming while the package was under active use and integration.

### 3. Internal naming / organization cleanup

We discussed whether naming like multiple different `models` files could be improved, and whether some files such as `space_models.py` should be reorganized later.

We intentionally deferred a broad reorganization until the package stabilizes further.

### 4. Shared spatial base package

Longer term, we may want a tiny shared base package for truly general spatial primitives like `SpaceDefinition`, but we explicitly chose not to extract that yet.

## Main tasks still to do

### High priority

- continue validating and stabilizing the weighted template update workflow
- clean up and document the transform layer semantics and examples
- continue testing transform operations on real registration outputs and template workflows

### Medium priority

- build HPC-ready registration runners (likely first in `lsfm_atlas_framework`)
- improve docs around template-update math and normalized vs output intensity spaces
- add more reusable examples for common lab workflows

### Later / optional

- revisit the `atlasspace` rename
- revisit package/file naming consistency
- consider future backend abstraction beyond ANTs if needed
- consider extracting shared spatial primitives into a tiny lower-level package

## Practical summary

Atlasbuilder is now in a useful and already integrated state. It is no longer just a repo idea or a planning scaffold. It is actively providing:

- registration infrastructure
- reusable spatial image utilities
- confidence-map and template-update logic
- a bridge into downstream workflow repos

The transform layer is now in place and working against real registration
outputs. The biggest recently active area has been stabilizing the
template-update math and then bringing transform semantics into alignment with
the validated registration behavior.
