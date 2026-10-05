# Release validation

Validation was performed on macOS arm64 with the locked Python 3.12 CPU
environment. The original implementation and publication-update sources were
frozen before editing. Their inspected files remained unchanged throughout
release preparation. Validation outputs and source snapshots are separate
from the publishable code.

## Checks completed

| Check | Result |
| --- | --- |
| Existing broad suite, before reducing tests | 1,093 passed; one CUDA check skipped; one deselected |
| Reduced release suite, with the verified bundle | 120 passed across 12 focused modules |
| Core/protein environment | Clean locked install and real library imports passed |
| LaMBO-2 environment | Locked Python 3.10 install and real LaMBO-2 import passed |
| Alternate main interpreter | Python 3.11 core install/imports passed |
| Fresh clone and built wheel | Separate environment, out-of-source imports, all 120 tests; the 1,340-cell configuration plan also passed |
| Artifact import and verification | All 2,630 files verified; corrupted inputs and conflicting existing files rejected |
| Local experiment planning | 1,340 cells; declared configurations, frozen checkpoints and labelled archives resolved |
| Exhaustive configuration comparison | All 1,340 cells across 73 task/method cohorts preserve scientific controls |
| Training planning | Four recipes validated; no training launched |

The compact suite covers installation and checksum boundaries, configuration
resolution, charged budgets, oracle isolation, proposal selection, kernels,
protein adapters, external algorithms, Sokoban solver/search behavior and
reporting. Provider checksum checks run before loading EvoDiff/ESM model
weights. All three protein weight files were subsequently downloaded and verified
by their native provider checksums and exact SHA256 hashes. See the
[one-seed real execution checks](smoke_validation.md) for the additional
checks and two runtime fixes found after the initial preparation.

The wheel checks used an empty source-search path and the fresh clone's
configurations. The sole installed command is `genturbo`; both asset manifests
are packaged. The complete artifact verification passed from the installed
wheel, and dry-run inspection created no experiment output. The tracked-file
audit found no generated figures, papers, data, checkpoints, private paths or
retired execution requirements.

## Scientific comparisons

The configuration audit compared every resolved cell against its appropriate
frozen source. Of 1,340 cells, 1,120 matched after output, path and provenance
normalization. The remaining 220 differed only in the LaMBO-2 environment-lock
path (100), Sokoban solver workers changing from four to one (100), or explicit
AdaLead alphabet/RNG/disabled-crossover defaults (20). No additional changes
to budgets, seeds, initial archives, generators, GP settings, acquisitions,
history, parents, candidates or oracle settings were found.

Reference and release implementations ran in separate Python processes under
the same environment. Imported module locations were checked to exclude
accidental use of the original checkout.

- Protein AdaLead and faithful publication Ehrlich AdaLead: matching proposals,
  predictor calls, RNG behavior and diagnostics over three seeds and two rounds.
- Ehrlich CTMC/GP search with a real frozen generator: matching 64 initial rows,
  16 paid search rows, two diagnostic rows and 16 proposal-trace rows.
- Sokoban direct methods (faithful AdaLead, Hamming and grid GA): matching short
  GP/solver runs, including selections, paid scores and solver outcomes.
- Sokoban sequence-policy GenTuRBO and independent generator: both used the
  frozen generator and verified seed-0 starts, 64 initial evaluations and four
  additional paid queries over two rounds. Generated pools, selections, RNG
  states, exact solver witnesses and archives matched.

These comparisons required exact discrete decisions and matched numerical
fields exactly. The allowed numerical tolerance was `rtol=1e-6, atol=1e-8`
only when decisions remained unchanged; no relaxation was needed for the
reported comparisons. Output locations, elapsed timing and additional release
provenance were excluded from equality. Historical timing measurements remain
preserved as scientific reference inputs.

The public dispatcher also completed independent short runs:

| Local path | Initial + new charged labels |
| --- | ---: |
| Ehrlich faithful AdaLead | 64 + 32 |
| Ehrlich CTMC + GP | 64 + 8 |
| AAV-M Hamming with the real CNN oracle | 128 + 8 |
| Sokoban Hamming with the exact solver | 64 + 1 |

The full forty-cell kernel discovery cohort was copied from frozen inputs and
successfully finalized into eligible-candidate tables, locality statistics and
trace-hash provenance. Those original frozen inputs were not modified.

## Numerical report replay

The complete reference report was regenerated without manuscript files.
Recomputed outputs matched all 84 protein Welch/Holm comparison rows, 32
primary comparison rows and 257 exact neighbourhood-count rows, with no
numerical differences. All main trajectory/figure inputs come from the
separately checksummed numerical records. The visitation/cone report consumes
the saved 100,000-draw numerical input; replay does not claim a new draw cohort.

## Scope and limits

Full published cohorts were not rerun. Linux CUDA installation/execution,
MPS numerical parity, full-length online LaMBO-2 search, full retraining,
complete static/timing sweeps and the full protein calibration replay remain
unverified. Short real EvoDiff, ESM, Kermut and LaMBO-2 runs now pass. Short parity runs establish behavior for
the tested inputs; they do not establish equivalence of every stochastic run
or hardware profile.

The bundle and checksum are available from the
[v0.1.0 release](https://github.com/Detriatis/GenTuRBO/releases/tag/v0.1.0).
New scientific cohorts remain outside these checks.

## Reductions

The original 111 test files were reduced to 12 focused modules after baseline
and parity checks. Configuration files were reduced from 486 JSON recipes to
89 files including their small manifest. Python source and local scripts were
reduced by about 50%; test code was reduced by about 93%. Research coordination, scheduler
launchers, manuscript integrations, unused molecular/GB1 workflows, toy
demonstrations and obsolete sweep/reporting interfaces were removed.

Scientific implementations and compatibility modules needed by the retained
experiments remain. Further simplification could replace the remaining
internal legacy protocol builders with fully resolved named templates and
split the large shared search engine along proposal/oracle/acquisition
boundaries. Either change should retain the same configuration and real-run
parity checks; neither is needed to run this release.
