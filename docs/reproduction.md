# Reproduction recipes

Run commands from the repository root after setup. All examples use the
`artifacts/` directory created by bundle import. Use absolute paths when
storing assets or outputs elsewhere. Specify `--repo-root .` when using an
installed wheel alongside a clone.

## Main experiments

`genturbo reproduce` resolves the complete configuration for every selected
cell and prints its inputs, output location and budget. Inspection is the
default; add `--run` to execute. Seeds accept values or ranges, such as
`--seeds 0 4 8` or `--seeds 0-19`. Unknown methods/tasks are errors.

| Suite | Tasks | Method identifiers |
| --- | --- | --- |
| `ehrlich` | `l256_q4_k8` | `unconditional`, `no_tr_no_gp`, `tr_no_gp`, `no_tr_gp`, `tr_gp`, `oracle_upper_bound`, `unconditional_gp`, `hamming_random`, `hamming_gp`, `adalead_official`, `bo_evo`, `lambo2`, `holo_ga` |
| `protein` | `gfp_medium`, `gfp_hard`, `aav_medium`, `aav_hard` | `kermut`, `tanimoto`, `hamming_gp`, `holo_ga`, `adalead`, `bo_evo`, `lambo2` |
| `sokoban` | `box_switches` | `genturbo`, `generator`, `adalead_official`, `grid_ga`, `hamming` |

The main cohort contains 260 Ehrlich, 560 protein and 100 Sokoban runs.
Iterative Ehrlich and Sokoban methods share 64 labelled initial designs;
protein methods use 128. The total budget is 2,000 charged labels per run.
Ehrlich unconditional sampling starts cold. Its oracle upper-bound policy
records privileged scores separately from the selected archive budget.

Protein recipes preserve the current effective settings, including GFP-M
fixed noise variance 0.001 and the final external-baseline settings. Original
labelled starts are loaded, not regenerated or rescored. Approved AAV splits
are accepted by their recorded hashes.

For example, run two AAV-M Hamming baselines:

```bash
genturbo reproduce protein --repo-root . --artifacts artifacts \
  --tasks aav_medium --methods hamming_gp --seeds 0-1 \
  --device cpu --output runs/reproduction --run
```

Run the five Sokoban methods for one seed:

```bash
genturbo reproduce sokoban --repo-root . --artifacts artifacts \
  --seeds 0 --device cpu --output runs/reproduction --run
```

Generation/oracle device choices are explicit; surrogate settings remain
those of the recipe. The default of one worker avoids local oversubscription.
`--threads` controls compute threads. GPU-heavy protein methods can be very
slow on CPU. There is no automatic fallback from an unavailable device.

The runner records resolved configurations, labelled starts, samples and
completion metadata beneath
`<output>/<suite>/<task>/<method>/seedN/`. A completed run must match its
declared budget. Reusing an output location with a changed configuration is
an error. Use a new output directory for a changed execution profile.

## Supporting analyses

`genturbo reproduce supporting --analysis STAGE` selects a named study.
Omitting `--analysis` selects all supporting studies.

| Stage | Reproduced work |
| --- | --- |
| `protein-components` | Four component variants × four tasks × 20 seeds (320 cells) |
| `protein-surrogate` | GFP-M learned-noise comparison, 20 seeds |
| `protein-calibration` | Five learned/five fixed GFP-M histories, held-out validation across five surrogate variants, then the frozen two-model test comparison |
| `kernel-datasets` | Forty discovery runs, candidate freezing and locality calibration |
| `kernel-static` | Static held-out kernel comparison on frozen eligible candidates |
| `kernel-timing` | Kernel timing benchmark; historical recorded timings remain separate reference inputs |
| `kernel-online` | Two conditions × four kernel arms × five seeds (40 cells), followed by trajectory aggregation |
| `visitation` | 100,000 frozen-generator draws and the local-neighbourhood mechanism figure |
| `counts` | Exact neighbourhood/support counts using the recorded scientific instance |

For example:

```bash
genturbo reproduce supporting --repo-root . --artifacts artifacts \
  --analysis visitation counts --output runs/reproduction --device cpu --run
```

The complete `all` plan contains 1,340 experiment cells plus analysis jobs.
Individual supporting stages may create prerequisite cells. The calibration
stage can run independently with its ten required search cells; in the
complete plan it reuses the main and surrogate cohorts.

Full-cohort aggregation is disabled when task, method or seed filters select
a partial experiment. Kernel static/timing stages need the completed
`kernel-datasets` runs in the same output location. For a full supporting
reproduction, run the unfiltered supporting plan. Retain measured wall times
when reporting a new timing experiment; do not substitute them for historical
timings in a replay of the published report.

## Reports

Replay reference numerical results:

```bash
genturbo report all --artifacts artifacts --output reports/reference
```

Rebuild from completed new runs:

```bash
genturbo report all --runs runs/reproduction --output reports/new
```

Reports consume numerical records and solver witnesses. They regenerate
figures, trajectories and statistical tables. No manuscript files are read.
The Sokoban primary endpoint is the best exact minimum-box-switch score;
the top-ten mean remains secondary. Protein fitness values retain their
recorded normalization and oracle-score precedence. The combined report
recomputes Welch comparisons and Holm corrections.

The destination must be empty or absent. Partial or inconsistent cohorts
are rejected rather than presented as publication results. `--reference`
can select a different numerical reference root containing
`figures/data/` and `figures/sokoban/`; supporting replay also needs its
visitation inputs or the artifact bundle.

## Preparing data

Reproduction uses the frozen bundle. The following commands are provided
for rebuilding data in a separate directory:

```bash
python scripts/download_gfp_aav_sources.py --root data/gfp_aav
python scripts/prepare_gfp_aav.py --root data/gfp_aav
python -m sokoban.prepare --download --output runs/data/boxoban
```

GFP/AAV preparation is not a replacement for the checksum-approved publication
splits. Keep newly prepared outputs separate from imported assets. Kermut
arrays are frozen in the bundle because the historical upstream source
revision was not recorded; `scripts/prepare_gfp_aav_kermut_resources.py --help`
describes preparation from an explicitly supplied source.

## Retraining generators

Retraining creates new model checkpoints; it is not required to reproduce
experiments using the frozen checkpoints. Start by inspecting the plan:

```bash
genturbo train ehrlich --repo-root . --artifacts artifacts \
  --output runs/retraining/ehrlich --device cpu --dry-run
```

Add `--run` to train. The named recipes are `ehrlich` (the main CTMC
generator), `ehrlich-short` (the shorter-sequence generator),
`sokoban` and `lambo2-ehrlich`. Ehrlich training stops at the recorded
18,000 updates. LaMBO-2 uses its isolated runtime and reward-free pretraining
contract. Sokoban requires the prepared Boxoban data and retains the recorded
training budget:

```bash
genturbo train sokoban --repo-root . --artifacts artifacts \
  --data-root runs/data/boxoban --output runs/retraining/sokoban \
  --device cuda --run
```

CPU/MPS Sokoban training uses float32; this is a different hardware/precision
profile from the recorded CUDA training. Model initialization and floating
point behavior can differ across devices and library builds. Do not overwrite
the frozen model inputs with newly trained checkpoints.
