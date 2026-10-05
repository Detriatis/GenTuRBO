# External sources and frozen inputs

Each subdirectory identifies an external implementation or dataset. `manifest.json`
drives the LaMBO-2 adapter installation and validates installed-source declarations
against both projects and lockfiles before setup. It records immutable runtime pins and distinguishes installed dependencies from
in-tree algorithm adaptations and data sources. The setup command installs runtime
Git dependencies at those exact commits. Optional checkouts belong in each
subdirectory's ignored `source/` folder.

The companion `GenTuRBO-artifacts.tar.gz` is distributed separately from code
through the [v0.1.0 release](https://github.com/Detriatis/GenTuRBO/releases/tag/v0.1.0).
`artifact-bundle.json` records the download URLs and bundle hash; `src/genturbo/artifact_manifest.json`
records every payload's relative path, group, size and SHA256. Import/verification
never substitutes newly generated data for frozen scientific inputs. See
[setup instructions](../docs/setup.md).

Original scientific payload bytes are preserved in the companion archive. Legacy
provenance embedded in those payloads describes the historical execution; it is
not needed as a local machine path for this release.
