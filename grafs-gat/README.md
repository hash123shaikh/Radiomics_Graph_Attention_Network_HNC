# GRAFS-GAT

**Outcome prediction in head-and-neck cancer (locoregional recurrence, distant metastasis, overall
survival) with a radiomic graph attention network, comparing feature-selection methods × graph
topologies for discrimination *and* reproducibility, validated with the design of Mateus et al. (2023).**

Builds on
- A spatial radiomic graph approach (Bae et al., *Radiology: Imaging Cancer* 2025): GTV + peritumoral supervoxel star graph, GAT, GTV read-out.
- Graph-FS (Moradmand et al., *Sci Rep* 2025;15:27995): label-free feature selection on a feature-similarity graph.
- Mateus et al., *Sci Rep* 13, 18176 (2023): training/validation on the 4 Canadian centres, external test on
  MAASTRO, 5-fold stratified CV and the Diamant et al. (2019) cohort split, outcome horizons.

## Quick start (reproduction)

| Step | Command | Details |
|---|---|---|
| Environment | `conda env create -f environment.yml` (or `docker build -t grafs-gat .`) | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| Installation test, no data | `make smoke` | synthetic data, ~2 min |
| Data | `python tools/organize_data.py ...` then `make check` | [DATA.md](DATA.md) |
| Earlier model release | `make verify` | v0.1.0-models (v3, 199-patient analysis) on MAASTRO, no training |
| Full study | `make study` | preprocessing v4, as in the manuscript |
| Manuscript assets | `make export && make paper` | every figure, table and number |

Paths and the preprocessing profile are set with environment variables (`GRAFS_DATA_DIR`,
`GRAFS_OUTPUT_DIR`, `GRAFS_PREPROCESS`), never by editing code.

## Research question

> Which feature-selection strategy gives node features that are both **predictive** and **reproducible**
> across centres when used in a spatial radiomic GAT, and does the graph topology matter?

| Axis | Options (`grafsgat/config.py`) |
|---|---|
| Feature selection (9) | `table_s3` (paper reference), `all` (no selection), `mrmr`, `lasso`, `boruta`, `rfe_rf`, `graph_fs_cc`, `graph_fs_spectral`, `graph_fs_hybrid` (Graph-FS representatives re-ranked by mRMR) |
| Graph topology (4) | `radiomic_star` (paper), `spatial_star`, `radiomic_knn`, `spatial_knn` |
| Inputs (2) | imaging only, imaging + clinical |
| Seeds | `BENCHMARK_SEEDS` (predictions are averaged over seeds) |
| Non-graph control | random forest on the same selected features, and a clinical-only RF |

Selection, scalers and encoders are fitted on **training patients only**. The configuration
is chosen on **validation AUC** and reported **once** on test. Methods are compared with paired
DeLong tests (Holm-corrected) against the paper reference (Table S3 + radiomic star) and against an RF on
the same features.

## Outcomes, cohorts and validation design

| Task | Outcome | Horizon (Mateus et al.) | HNPET train/val: patients (events) | MAASTRO external test |
|---|---|---|---|---|
| `LR` | locoregional recurrence | 2 years | 276 (30) | 121 (30) |
| `DM` | distant metastasis | 2 years | 276 (27) | 109 (7) |
| `OS` | death from any cause | 4 years (median follow-up) | 164 (44) | ≈135 (≈45) |

Label = event within the horizon; patients followed for less than the horizon without an event are
excluded. HNPET = full TCIA Head-Neck-PET-CT, 298 patients (CHUM, CHUS, HGJ, HMR; DICOM CT + primary-GTV RTSTRUCT in
`data/Head-Neck-PET-CT`, GTV names in `gtv_primary_rois.csv`, outcomes from `INFOclinical_HN_Version2_30may2018.xlsx`); MAASTRO = HN1. Exact counts per run are in
`outputs/results/cohort_summary.csv`.

| `--mode` | Train / validation (HNPET) | Test |
|---|---|---|
| `MATEUS_CV` (primary) | stratified 5-fold CV; one full run per fold in `fold1..fold5/` | MAASTRO, every fold |
| `MATEUS_COHORT` | by centre: train HGJ + CHUS, validate HMR + CHUM (Diamant et al.) | MAASTRO |

In `MATEUS_CV`, feature selection, scalers, encoders and models are refitted inside every fold. The
mode folder holds the cross-fold summary:
- `val_auc` is the pooled out-of-fold AUC; every training patient is validated exactly once, and the
  configuration is chosen on this value.
- `test_auc` is the AUC of the fold ensemble (probabilities averaged over folds and seeds) on MAASTRO,
  with a 95% bootstrap CI; `test_auc_fold_min/max` give the single-fold range reported by Mateus et al.
- `cv_*` columns measure the stability of feature selection across the folds.

Checkpoint rule: lowest validation loss (Bae et al. 2025, primary). `--epoch_rule auc_gap` applies Mateus
et al.'s rule instead (highest validation AUC among epochs with train AUC − validation AUC ≤ 0.05, the
threshold raised in steps of 0.05 until an epoch qualifies) and writes to `<MODE>_<TASK>_aucgap/`.

The Table S3 reference features exist only for LR and DM; for OS the reference GAT is mRMR + radiomic star,
and the GAT uses the LR hyper-parameters (Table S2 has no OS entry).

## Pipeline

```bash
python check_setup.py                       # environment + data check
./run_mateus_study.sh                       # everything: LR, DM, OS x both designs x both checkpoint rules
python export_results.py                    # (re)collect results; nothing is retrained
```

| Stage | What it does | Output |
|---|---|---|
| `preprocess` | CT + primary GTV → 1 mm, 5 cm box, SLIC supervoxels | `outputs/cache/preprocessed/` |
| `extract` | 93 PyRadiomics features for the GTV and every supervoxel | `outputs/cache/features/` |
| `select` | every FS method on the training split | `runs/<run>/selection/*.json` + figures |
| `stability` | bootstrap stability + cross-centre reproducibility | `selection/stability_*.csv` + figures |
| `benchmark` | graphs, GAT grid, RF baselines | `graphs/`, `models/` (weights), `results/`, `predictions/` |
| `summary` | seed-ensembled AUC + CI, DeLong/Holm, figures | `results_summary.csv`, `figures/`, **`report.html`** |

Every stage is cached: models, metrics and predictions that exist are not recomputed (use `--force` to redo).

```bash
python main.py --mode MATEUS_CV --task DM --from_stage select          # 5 folds + cross-fold summary
python main.py --mode MATEUS_CV --task OS --from_stage select --fold 3 # a single fold
python main.py --mode MATEUS_COHORT --task LR --from_stage select --epoch_rule auc_gap
python main.py --mode MATEUS_CV --task LR --stage benchmark --methods graph_fs_cc mrmr --topologies radiomic_star --seeds 42
```

## Study results (`outputs/results/`, written by `export_results.py`)

| File | Content |
|---|---|
| `main_results.csv` | per task × design × checkpoint rule × inputs: chosen GAT, reference GAT, best RF, clinical-only RF → validation AUC, external test AUC [95% CI], single-fold range, Holm-corrected DeLong p vs reference |
| `all_configurations.csv` | every configuration of every run |
| `feature_selection_stability.csv` | stability of every FS method (across CV folds, bootstrap, centres) |
| `cohort_summary.csv` | patients and events per task and cohort |
| `predictions/*.csv` | per-patient probabilities of the chosen models (out-of-fold validation + external test, each fold's probability) |
| `figures/roc_<TASK>.png` | ROC curves on MAASTRO (chosen GAT with 95% bootstrap band, reference, RF, clinical) with AUC [CI] |
| `figures/roc_cv_folds_<TASK>.png` | ROC of every CV fold + mean ROC ± 1 SD |
| `figures/auc_overview.png` | external test AUC [95% CI] across tasks and designs |

## Saved models and inference (`predict.py`)

`outputs/final_models/<DESIGN>_<TASK>[_aucgap]/<imaging|imaging+clinical>/` holds the chosen configuration's
checkpoints (5 folds × 3 seeds for CV) with `model_card.json`. Every checkpoint is self-contained (weights,
selected features, GTV scaler, clinical encoder, threshold, topology), so no retraining is needed:

```bash
python predict.py --task LR --dataset MAASTRO                                   # chosen CV model, whole cohort
python predict.py --task OS --design MATEUS_COHORT --inputs imaging+clinical --dataset MAASTRO
python predict.py --task DM --config mrmr__spatial_star --dataset MAASTRO --patients HN1004 HN1006
python predict.py --task LR --models outputs/final_models/MATEUS_CV_LR/imaging --dataset MAASTRO
python predict.py --task LR --ct new/CT.nii.gz --mask new/GTV.nii.gz --id NEW001   # a new patient
```

Output (`outputs/predictions/`): ensemble probability, its spread across members, high/low risk at the mean
validation Youden threshold, the label when known, and whether the patient was used for training.

## Per-run figures (`outputs/runs/<MODE>_<TASK>/[foldk/]figures/`)

| Figure | Shows |
|---|---|
| `selection/feature_graph_graph_fs_*.png` | Graph-FS feature-similarity graph; features coloured by class, selected representatives ringed |
| `selection/selected_features_matrix.png` | which features each method picked |
| `selection/method_agreement_jaccard.png` | overlap between methods |
| `selection/stability_bootstrap.png`, `stability_centres.png` | Jaccard, Dice, Kuncheva, Kendall's W per method |
| `selection/selection_frequency_bootstrap.png` | how often each feature is re-selected |
| `results/auc_heatmap_*.png` | test/val AUC, method × topology |
| `results/forest_test_auc_*.png` | every configuration with 95% CI |
| `results/stability_vs_auc_*.png` | the key figure: reproducibility vs discrimination |
| `results/roc_overlay_*.png` | chosen GAT vs paper reference vs best RF vs clinical |
| `patient_graphs/*.png` | a test patient's graph on the CT slice, coloured by GAT attention, plus the graph itself |
| `training/*.png` | loss/AUC curves of the chosen model |

Everything is collected in `report.html` (one self-contained file).

## Code layout

```
main.py                     CLI (all stages; --preprocess v3|v4)
run_mateus_study.sh         the full study (all tasks, both designs, both checkpoint rules) + export
export_results.py           study-level CSVs, ROC figures, final model copies
predict.py                  inference with saved models (cohort patients or new NIfTI files)
verify_release.py           re-apply the released models and compare with the reported external AUCs
check_setup.py              environment / data check (patient and event counts vs the paper)
smoke_test.py               every stage on synthetic data (installation test)
tools/organize_data.py      arrange the TCIA DICOM downloads into the expected layout
Makefile                    make smoke | check | verify | study | export | paper | docker
environment.yml, requirements.txt, requirements-lock.txt, Dockerfile    environments
DATA.md, REPRODUCIBILITY.md data instructions; reproduction levels and what has been verified
grafsgat/
  config.py                 cohorts, modes, FS + topology grid, GAT (Table S2), run layout
  stats.py                  DeLong test, Holm correction
  utils.py                  seeding, metrics, bootstrap CI
  data/        clinical.py (labels + encodings), loader.py (CT/GTV), preprocessing.py,
               supervoxels.py, features.py (PyRadiomics + caches), splits.py (CV / cohort split)
  selection/   methods.py (9 selectors, Graph-FS), stability.py (Jaccard/Dice/Kuncheva/Kendall W),
               runner.py (select + stability stages)
  graphs/      builder.py (4 topologies), dataset.py (datasets, class-balanced sampler)
  models/      gat.py (GRAFS-GAT network)
  engine/      train.py, evaluate.py (+ attention), baselines.py (RF)
  experiments/ benchmark.py (grid, summary), cv.py (cross-fold summary), export.py (study results)
  inference.py              load checkpoints, build graphs, ensemble prediction
  provenance.py             provenance.json (git commit, versions, settings) for every run
  viz/         style, feature_graph, selection_plots, result_plots, patient_graph, report, paper_figure
```

## Design notes / limitations to report

- Graph-FS edge rule: |Pearson r| > mean + 0.3·SD; isolated features are linked to their most similar
  feature; CC representative = highest degree centrality; spectral = unnormalised Laplacian + k-means,
  representative = closest to its cluster centroid.
- Default `FS_K = 6` features per method (fair comparison); `--k native` uses each method's own size (clipped to 3–20).
- The radiomics are computed with one PyRadiomics setting, so reproducibility is measured across bootstrap
  resamples and centres, not across extraction parameters as in Moradmand et al.
- Event counts are small (13 LR, 12 DM, 17 OS events for training; 7 DM events in MAASTRO), so CIs are wide.
  Report CIs and corrected p values, and avoid ranking methods on point estimates. DM in particular has too
  few external events for firm conclusions.
