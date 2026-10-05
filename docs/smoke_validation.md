# One-seed real execution checks

On 5 October 2026, the release was exercised with real frozen inputs, real
model weights and real oracles on CPU, using seed 0. No mock generators or
oracles were substituted. All final checks listed below passed.

These are deliberately small execution checks. They change search budgets,
candidate counts and, where needed, the number of saved initial records.
They do not reproduce the publication's scores or establish statistical
performance. Canonical experiment configurations were not changed.

| Coverage | Reduced execution |
| --- | --- |
| All 13 Ehrlich policies | Normally 64 saved labels plus two new labels; Holo used four saved plus four new |
| All seven protein methods | AAV-Medium, one update; includes actual EvoDiff, ESM2, Kermut and online LaMBO-2 |
| Four protein component variants | Random-time/random-selection, trust-region/random-selection, and both random-time GP variants |
| All four protein tasks | Hamming + GP/EI with 128 original labels plus two real CNN evaluations per task |
| All five Sokoban methods | 64 original labelled boards plus two paid evaluations; actual generator, GP and exact solver |
| All four online-kernel arms | Shorter Ehrlich condition, 64 saved labels plus two paid evaluations |
| Static kernel and timing pipelines | One-hot Tanimoto, eight training/four test examples, one fit/repeat and materialized outputs |
| All five calibration variants | Eight stored training labels and four real validation probes |
| Locality/mechanism | 32 real frozen-generator draws; exact 24,055-state local support and exact graph counts |

Protein generator checks used eight unchanged initial labels, four candidates,
two denoising steps and two new oracle evaluations. Protein LaMBO-2 used
16 initial labels and two new evaluations. Protein Holo used eight initial
labels and eight new evaluations. Small Holo populations used an explicitly
adjusted survival fraction to retain two survivors. Candidate-pool and batch
settings remained consistent with each algorithm's required invariants.

The oracle upper-bound Ehrlich check separately verified its 66-label
selected-archive budget, 72 physical evaluations and six privileged scores.
Other checks verified charged-label totals, completed manifests and saved
outputs. Protein generator outputs contained two finite, real CNN scores.
Original labelled records were preserved without rescoring.

Most ordinary search checks took approximately 1–4 seconds. Online LaMBO-2
took approximately 22 seconds for AAV and 40 seconds for Ehrlich. Protein
generator/embedding checks took 46–79 seconds each, including model loading
and checksum checks. Each experiment had a 120-second limit. Model downloads
were preparation overhead, separate from these timings.

## Issues found and corrected

- Holo's shared oracle-accounting helper had been removed during pruning.
  The seven required functions were restored from the frozen implementation,
  and a prepaid-budget regression now runs the real baseline.
- ESM's version-specific S3 download URL rejected anonymous downloads.
  Prefetch now uses the official public URL and verifies the exact recorded
  bytes. All three downloaded model files passed provider checksums and
  newly recorded SHA256 checks.
- An unused protein-pretraining entry point still imported a removed module.
  It was removed; retained LaMBO-2 training uses the supported sequence module.

Earlier failed smoke attempts caused by incompatible tiny population/batch
overrides are retained in the local evidence alongside corrected final runs.
They were changes to the validation harness, not changes to canonical
scientific settings.

## Verification and limits

After these fixes, all 120 compact tests passed against the rebuilt installed
wheel. Every retained importable source module was checked against that wheel,
and all cached protein weights passed the new SHA256 and provider guards.

Raw logs, resolved overrides and numerical outputs are kept separately in the
local ignored `runs/validation-20261005/` directory. They are not publication
results and are not tracked in Git. Downloaded weights are also outside Git.

Full cohorts, long-run convergence, GPU execution, hardware parity and
retraining remain untested by these checks. Protein generator methods were
exercised on AAV-Medium; all four protein tasks exercised their real task
adapters and CNN oracle through Hamming search. The static/timing smoke covers
one kernel rather than the complete benchmark grid.
