# Reproducing the study

There are four levels of reproduction, from minutes to days. Each one is a single command, run from
`grafs-gat/`.

| Level | What it shows | Needs | Time | Command |
|---|---|---|---|---|
| **0. Installation** | every pipeline stage runs on this machine | nothing (synthetic data) | ~2 min, CPU | `make smoke` |
| **1. Published external results** | the released models give the reported MAASTRO AUCs | MAASTRO data | ~1–2 h for MAASTRO features (once), then minutes | `make verify` |
| **2. Full study** | the whole study retrained from the raw images | both cohorts, GPU | several GPU-days | `make study` |
| **3. Manuscript** | every figure, table and number of the paper from the results | results of level 2 | ~10 min | `make export && make paper` |

## Environment

Choose one of three options:

```bash
# (a) conda, exact versions
conda env create -f environment.yml && conda activate GNNvenv

# (b) pip into any Python 3.8 environment
pip install numpy==1.24.3 "setuptools>=65" && pip install -r requirements-lock.txt

# (c) Docker (no local Python needed)
docker build -t grafs-gat .
docker run --rm grafs-gat                                                     # level 0
docker run --rm --gpus all -v $PWD/data:/data -v $PWD/outputs:/outputs grafs-gat \
       python verify_release.py --download                                     # level 1
```

The three files differ in strictness:
- `requirements.txt`: the direct dependencies, with pinned versions.
- `requirements-lock.txt`: every package at the exact version of the environment that produced the results (Python 3.8.20, PyTorch 2.0.1 + CUDA 11.7, Linux).
- `environment.yml`: conda Python 3.8 plus that lock file.

## Data

See [DATA.md](DATA.md). `python check_setup.py` compares your patient and event counts with the paper.

## Preprocessing profile

The manuscript results use **preprocessing v4** (body mask + soft-tissue window before SLIC), the default
of every script. **v3** (SLIC on the whole box with min–max scaling) is kept only for the models of release
`v0.1.0-models`, which came from an earlier analysis of the 199-patient HECKTOR image set and are superseded.
You select the profile without editing code:

```bash
python main.py ... --preprocess v3          # or: export GRAFS_PREPROCESS=v3
python predict.py ... --preprocess v3
```

Cached files record their profile, so files of a different profile are recomputed, never mixed. If no
cached features match the selected profile, the pipeline stops with an explanation instead of silently
creating a new data split. Give a v3 run its own output folder so that it does not replace the v4 cache
(`make verify-v0` does this automatically with `outputs_verify_v3/`):

```bash
GRAFS_PREPROCESS=v3 GRAFS_OUTPUT_DIR=outputs_v3 python predict.py ...
```

## Current study run (17–18 Nov 2026)

Complete TCIA Head-Neck-PET-CT collection (298 patients, DICOM CT of the PET/CT + RTSTRUCT) and MAASTRO
(137), preprocessing v4: preprocessing/extraction 1 h 38 min with `N_JOBS=4`, all 12 runs (3 endpoints ×
2 designs × 2 checkpoint rules) 12 h, no errors. Patient and event counts equal Table 1 of Mateus et al.
(LR 276/30, DM 276/27, OS 164/44). With `N_JOBS=12`, preprocessing exhausted the 62 GB of RAM and the
desktop session was killed by systemd-oomd; use `N_JOBS=4` on a 64 GB machine.

## Release v1.0.0-models (18 Nov 2026)

`verify_release.py` on the packaged release zip (`grafs-gat_final_models_v4.zip`, 24 ensembles, 216 checkpoints):
**24/24** ensembles reproduce their reported external AUC to 4 decimals (e.g. LR, 5-fold CV, imaging: 0.5469).

## What has been verified (15 Nov 2026; earlier v3 analysis and release v0.1.0-models)

| Check | Result |
|---|---|
| v3 profile re-run from raw images (CHUM-001, HN1004) vs the archived v3 cache | CT, GTV and supervoxel arrays identical; 93 features identical to within 10⁻¹⁵ |
| `verify_release.py`: 24 released ensembles (3 endpoints × 2 designs × 2 checkpoint rules × 2 inputs) on MAASTRO | **24/24** reproduce the reported external AUC to 4 decimals |
| `make_paper_assets.py` on the study results | all 218 numbers and all LaTeX tables byte-identical; figures identical (PDFs differ only in their creation date) |
| `main.py --stage select` (LR, CV fold 1) re-run from the cached features | data split and selected features (mRMR, spectral Graph-FS, LASSO, Boruta) identical to the study run |
| `predict.py --models <release> --preprocess v3` (LR, cohort split, imaging + clinical) | external AUC 0.714, as reported (0.71) |
| `smoke_test.py` on CPU | 4/4 stages pass |
| Docker image (`python:3.8-slim` + `requirements-lock.txt`, no torch-scatter/sparse/cluster), CPU only | smoke test 4/4; `verify_release.py --download` **24/24**, identical AUCs to the GPU workstation |
| `check_setup.py` | patient and event counts equal Table 1 |
| `tools/organize_data.py` on simulated TCIA/NBIA and HECKTOR layouts | loader reads the arranged data correctly |

## Determinism

- **Splits:** stratified with a fixed seed (`RANDOM_SEED = 42`) and saved in each run's `split.json`.
  The released models include them.
- **Training:** every configuration is trained with seeds 42, 43 and 44, and the probabilities are
  averaged. cuDNN runs in deterministic mode.
- **GPU caveat:** GPU scatter operations in PyTorch Geometric are not bit-deterministic. Retraining on a
  different GPU or CPU can therefore change individual probabilities slightly, and occasionally which
  configuration is chosen on validation. The released models remove this source of variation for
  level 1.
- **Provenance:** every run writes `provenance.json` next to its results. It records the git commit,
  command, package versions, device, preprocessing profile and all key settings.

## Where results are written

`$GRAFS_OUTPUT_DIR` (default `grafs-gat/outputs/`):

```
cache/preprocessed/, cache/features/        per-patient preprocessing and radiomic features
runs/<MODE>_<TASK>[_aucgap]/                every model, graph, prediction, figure, report.html, provenance.json
results/                                    study-level CSVs and figures (export_results.py)
final_models/                               chosen ensembles + model cards (what the GitHub release contains)
```
