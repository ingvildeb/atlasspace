# Changelog

All notable changes to `atlasspace` will be documented in this file.

The project follows semantic versioning. Before `1.0.0`, minor releases may
include intentional API or data-contract changes documented here.

## [0.2.0] - 2026-08-28

This is the first tagged `atlasspace` release. Earlier development installs
reported version `0.1.0` but were not tied to an immutable release tag.

### Added

- A versioned `registration_result.json` manifest (`schema_version = 1`) as the
  canonical machine-readable registration result.
- Public helpers for loading registration results and migrating completed
  legacy output folders that contain `registration_summary.txt`.
- Configurable registration `output_root` support.
- Optional fixed-image padding for registration preprocessing.
- Installable TOML job-spec templates and registration preset resources.

### Changed

- Registration consumers should use `registration_result.json` rather than
  parsing the legacy text summary.
- The manifest filename and schema version are public constants under
  `atlasspace.registration` so downstream packages can perform capability
  checks before launching jobs.

### Compatibility

- Existing successful registrations do not need to be rerun solely to adopt
  the JSON contract. Use `migrate_legacy_registration_output()` to generate a
  validated manifest from supported legacy outputs.
