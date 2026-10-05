# Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
python scripts/setup.py --profile cpu --dev \
  --artifacts /path/to/GenTuRBO-artifacts.tar.gz
```

The command creates `.venv` from the committed lock, checks core imports, and
verifies/imports the separately distributed artifact bundle into `artifacts/`.
The CPU profile supports Linux CPU and Apple Silicon CPU/MPS. Python 3.12 is
the default; `--python 3.11` is supported. No scheduler is required.

For all protein experiments and LaMBO-2:

```bash
python scripts/setup.py --profile cpu --protein --lambo2 --dev \
  --artifacts /path/to/GenTuRBO-artifacts.tar.gz
```

Use `--profile cuda128` for Linux CUDA 12.8. This selects Torch wheels without
changing scientific protocol settings or automatically changing experiment
devices. Historical configurations retain their recorded devices; a device
override constitutes a different execution profile.

LaMBO-2 requires a separate locked Python 3.10 environment at `.venv-lambo2`.
One setup command creates both environments. Its poli-baselines adapter is
installed at the exact recorded commit with `--no-deps` after its locked
dependency closure, avoiding upstream's unpinned poli Git requirement.
`GENTURBO_LAMBO2_PYTHON` can point to an equivalent separately prepared runtime.

Add `--prefetch-models` with `--protein` to download EvoDiff D3PM uniform 38M
and ESM2 `esm2_t33_650M_UR50D`, including contact-regression weights, before a
run. These large third-party weights are not included in the environment or
companion archive. `TORCH_HOME` chooses their cache. Model identifiers and
runtime commits are pinned. The prefetch step verifies exact SHA256 hashes
for all three files before model loading, together with native provider
checksums: immutable Zenodo MD5 for EvoDiff, the recorded S3 multipart
checksum for ESM2, and SHA256 for regression weights. The ESM2 download uses
the official public URL; byte hashes enforce the recorded object version.
Every successful download also records a local SHA256 receipt. The packaged
`src/genturbo/model_weights.json` contains sizes, URLs, versions and hashes.
To check an existing cache without downloads, use
`python -m genturbo.model_weights`. The real EvoDiff and ESM loaders also
verify exactly the weights they use before loading; they fail with the prefetch
instruction when absent or mismatched. The release pins EvoDiff uniform 38M
and ESM2 t33 650M only. Other upstream model variants are not reproduction
profiles. Hamming and plain CNN-oracle runs do not require these weight caches.

## Frozen inputs

The companion archive contains frozen inputs and numeric reference outputs,
with no rendered figures or paper source. Download it and its checksum from
the [v0.1.0 release](https://github.com/Detriatis/GenTuRBO/releases/tag/v0.1.0),
then verify the downloaded bundle:

```bash
shasum -a 256 -c GenTuRBO-artifacts.tar.gz.sha256
```

Supply the downloaded bundle explicitly to setup or the importer below.
`external/artifact-bundle.json` records download URLs, the archive hash and
size. `src/genturbo/artifact_manifest.json` records every payload's relative
path, group, size and SHA256.

```bash
.venv/bin/python -m genturbo.assets import /path/to/GenTuRBO-artifacts.tar.gz \
  --root /path/to/artifacts
.venv/bin/python -m genturbo.assets verify --root /path/to/artifacts
```

Give experiment/report commands this same directory using `--artifacts`.
Verification can select `--group ehrlich`, `--group protein`, `--group sokoban`
or `--group reference`. Import validates every member/checksum before placing
files. Identical files may be re-imported; different existing files are never
overwritten. Allow space for staging and the final unpacked archive.

| Bundle path | Contents |
|---|---|
| `benchmarks/gfp_aav/` | Source data, CNN oracles, approved task splits and original labels |
| `runs/train/` | Frozen Ehrlich and Sokoban generators |
| `runs/ehrlich_kernel_v1/datasets/` | Supplementary datasets and shared Ehrlich initial archives |
| `runs/ehrlich_sequence_baselines_v2/pretraining/` | Ehrlich LaMBO-2 checkpoint/pretraining records |
| `runs/ehrlich_radial_search_v1/` | Frozen locality inputs and intro cone numeric data |
| `runs/protein_design_v1/resources/kermut/` | Exact GFP/AAV Kermut arrays |
| `runs/data/sokoban_initial/runN/` | Per-seed configuration, provenance and labelled initial boards |
| `results/` | Numeric tables used by retained reporting paths |
| `reference/paper/figures/` | Numeric inputs used by current paper figures |
| `reference/ehrlich_adalead_official/` | Current faithful AdaLead cohort |
| `reference/sokoban/` | Current five-method comparison and per-cell witnesses |

Payload bytes are preserved, including embedded historical provenance strings,
except for one provenance record relabelled together with its results folder.
Paths in that historical metadata are not required local locations; execution
uses the selected artifact root. Do not replace frozen labels by rescoring.
Approved AAV task splits retain their exact checksums despite the documented
historical row-count discrepancy. Kermut's original source archive revision
was not recorded; use the exact bundled prepared arrays instead of an
unverified latest data download.

## Check installation

```bash
.venv/bin/python -m pytest
.venv/bin/genturbo --help
```

`--dev` supplies pytest. See `external/manifest.json` and the named external
folders for immutable revisions, setup instructions and adapted-source
attribution. Optional source checkouts belong in ignored `source/` folders.
Setup validates EvoDiff, poli and ESM versions and source URLs against the
shared external manifest and both lockfiles before installation. The LaMBO-2
adapter install URL is constructed from that same manifest. Any disagreement
stops setup; setup never regenerates environment locks. On Windows replace `.venv/bin/`
with `.venv/Scripts/`.
