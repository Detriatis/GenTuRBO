# GenTuRBO

Code and local recipes for reproducing GenTuRBO's generator-guided optimisation
experiments. The Python distribution is `GenTuRBO`; imports and commands use
`genturbo`.

| Suite | Publication cohort |
| --- | --- |
| Ehrlich | 13 policies × 20 seeds |
| Protein | 7 methods × GFP/AAV Medium/Hard × 20 seeds |
| Sokoban | 5 methods × 20 seeds |

## Set up locally

Download the [asset bundle and checksum](https://github.com/Detriatis/GenTuRBO/releases/tag/v0.1.0)
(approximately 304 MB) before running setup.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from this checkout:

```bash
python scripts/setup.py --profile cpu --protein --lambo2 --dev \
  --artifacts /path/to/GenTuRBO-artifacts.tar.gz
source .venv/bin/activate
```

This creates the locked Python 3.12 environment and isolated Python 3.10
LaMBO-2 environment. Use `--profile cuda128` on a local Linux CUDA machine.
Omit `--protein --lambo2` when those methods are not needed. Protein model
weights can be downloaded and verified with `--protein --prefetch-models`.
See [setup and asset verification](docs/setup.md) and [external sources](external/README.md).

The matching asset bundle is `GenTuRBO-artifacts.tar.gz`; use its accompanying
SHA-256 checksum. Download the [bundle](https://github.com/Detriatis/GenTuRBO/releases/download/v0.1.0/GenTuRBO-artifacts.tar.gz) and
[SHA-256 checksum](https://github.com/Detriatis/GenTuRBO/releases/download/v0.1.0/GenTuRBO-artifacts.tar.gz.sha256)
from the v0.1.0 release, then import the local bundle during setup.

## Reproduce and report

Inspect the complete resolved experiment plan without running experiments:

```bash
genturbo reproduce all --repo-root . --artifacts artifacts \
  --output runs/reproduction --device cpu --dry-run
```

Run a selected experiment, preserving the full scientific budget:

```bash
genturbo reproduce ehrlich --repo-root . --artifacts artifacts \
  --output runs/reproduction --methods tr_gp --seeds 0 --device cpu --run
```

Generate the publication figures and statistical tables from the bundled
reference numerical results, without rerunning the full cohorts:

```bash
genturbo report all --artifacts artifacts --output reports/reference
```

Generate a report from a newly completed suite:

```bash
genturbo report protein --runs runs/reproduction --output reports/protein
```

Execution is sequential by default, with one worker and one compute thread.
Without `--run`, reproduction only prints a plan. Device overrides are explicit:
omitting `--device` preserves recorded devices and raises an error if one is
unavailable. CPU/MPS/CUDA results are not assumed to be numerically identical.
Partial experiments can be inspected, but publication reports require complete
cohorts and exact budgets.

See [reproduction recipes and retraining](docs/reproduction.md) for all method
names, supporting analyses and preparation commands, and
[validation](docs/validation.md) and [one-seed execution checks](docs/smoke_validation.md)
for the checks performed and their limits.

## Check the installation

```bash
GENTURBO_ARTIFACTS="$PWD/artifacts" python -m pytest
```

The artifact variable enables the reference-report replay check. Core tests do
not require the bundle. The compact suite covers configuration integrity,
budgets, oracle isolation, search decisions, protein adapters, external
baselines, Sokoban and reporting.

## Layout

- `src/genturbo/`: public commands, experiment recipes, adapters and reporting.
- Other `src/` modules: scientific implementations; established paths such as
  `ehrlich` are retained for serialized scientific objects.
- `configs/`: named experiment bases and small overrides.
- `external/`: immutable source references, attribution and asset manifests.
- `environments/lambo2/`: the isolated Python 3.10 environment lock.
- `scripts/`: local setup, data preparation and supporting analyses.
- `tests/`: focused release checks.

Outputs, downloaded source checkouts, data and model files remain ignored.
Configurations with `base_config` use whole-value replacement for child keys,
including nested dictionaries and lists; the supplied loaders resolve them.

Code is licensed under [GPL-3.0-or-later](license.txt). External code, data and
model weights retain their upstream terms listed in `external/`.
