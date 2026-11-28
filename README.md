# Graph neural networks on primary and nodal tumour radiomics for outcome prediction in head and neck cancer

Code and manuscript of a multicentre external validation study. It asks whether **patient-similarity
graph neural networks**, built on CT radiomic features, can predict recurrence, metastasis and death
after radiotherapy for head and neck cancer as well as a published deep learning model. It also asks
**what actually drives their performance**: the patient graph, or how the tumour is described.

<p align="center">
  <img src="manuscript/figures/Fig2_architecture.png" width="100%" alt="Overview of the method: whole-tumour radiomic features, inductive patient graph, two message-passing layers, ten-seed ensemble, external evaluation, and the edgeless control">
</p>

**Contents:** [The study in brief](#the-study-in-brief) ·
[Quick start](#quick-start) · [Step by step](#step-by-step) ·
[What to expect](#what-to-expect) · [Reading the results](#reading-the-results) ·
[Repository layout](#repository-layout) · [Reproducibility](#reproducibility)

---

## The study in brief

### Background
Radiotherapy for head and neck squamous cell carcinoma (HNSCC) often fails locally, at distant sites
or both, usually within two years. A reliable pretreatment estimate of each patient's risk could guide
treatment intensification or de-escalation. Mateus et al. (*Sci Rep* 2023) published a convolutional
neural network (CNN) for this task. They also made public a **benchmark protocol**: fixed patients,
endpoints and validation designs on which other methods can be compared.

Graph neural networks are attractive for small cohorts. In a **patient-similarity (population) graph**
every patient is a node, and edges connect patients with similar features. A network such as GCN,
GraphSAGE or GAT then predicts each patient's outcome from the patient's own features and those of
the neighbours. Whether the graph adds anything is rarely tested, because the same network without
edges is seldom reported.

### Questions
1. Can a graph neural network on CT radiomics match the published CNN on its own benchmark?
2. Does it matter which tumour the radiomic features describe: the primary tumour only, or the
   primary tumour together with the involved lymph nodes (the **whole tumour**)?
3. Does the patient graph itself help, compared with the identical network without edges?

### Design

| | |
|---|---|
| **Development cohort** | Head-Neck-PET-CT (TCIA), four Canadian centres (CHUM, CHUS, HGJ, HMR), 294 patients with CT and delineations |
| **External test cohort** | HN1 (TCIA), MAASTRO Clinic, the Netherlands, 137 patients; used for testing only |
| **Endpoints** | Locoregional recurrence (LR) and distant metastasis (DM) within 2 years; death (OS) within 4 years; each binary, as in Mateus et al. |
| **Validation designs** | (a) stratified 5-fold cross-validation on the Canadian patients; (b) cohort split: train HGJ + CHUS, validate HMR + CHUM. MAASTRO labels are never used for training or model selection. |
| **Tumour definitions** | Primary GTV only, or whole tumour = union of the primary and all nodal GTVs |
| **Features** | PyRadiomics, 1 mm voxels, bin width 25 HU: 479 whole-tumour features (14 shape, 93 original, 372 LoG); 1223 primary-GTV features (the same plus 744 wavelet features) |
| **Inputs** | imaging only, or imaging + the clinical variables of the CNN study (T and N category, GTV volume bins; optionally nodal-burden variables) |
| **Models** | GCN, GraphSAGE and GAT on an inductive k-nearest-neighbour patient graph (test patients receive edges from training patients but send none), and an **MLP with identical layers and no edges** as control |
| **Model choice** | a grid of feature sets, selectors (mRMR, LASSO, univariate filter), feature counts and neighbourhood sizes; the configuration with the highest **validation** AUC is chosen; every model is a 10-seed ensemble |
| **Additional analyses** | 2×2 ablation (tumour definition × clinical variables); paired bootstrap graph vs MLP; matched comparison of every graph configuration with the MLP on the same features; retraining with 10 fresh seed ensembles; conventional classifiers (logistic regression, random forest) |

### Main findings
* With whole-tumour features, the graph networks reached external AUCs of **0.59–0.63** (LR),
  **0.85–0.87** (DM) and **0.70–0.73** (OS). They exceeded the CNN in **9 of 12** endpoint × design
  × input settings, against 6 of 12 with primary-GTV features.
* The gain came from the **tumour definition**: including the nodal GTVs in the radiomic region raised
  the DM AUC by up to 0.21. Nodal burden entered as clinical variables added nothing once the nodes
  were part of the region.
* The **patient graph added nothing**. The MLP without edges exceeded the CNN in 8 of 12 settings,
  and only 1 of 18 paired differences was significant, in favour of the MLP. With identical features
  the graph network was better on MAASTRO in only 29% of 1944 matched comparisons.
* For LR, a clinical model including age and HPV status outperformed every imaging model.

---

## Quick start

```bash
git clone https://github.com/hash123shaikh/Radiomics_Graph_Attention_Network_HNC.git
cd Radiomics_Graph_Attention_Network_HNC
conda env create -f environment.yml && conda activate hnc-popgraph   # 1. install
#   2. download the two TCIA collections and arrange them  (see "Step by step", 2)
python scripts/check_data.py                                         # 3. check: all lines must say OK
./run_study.sh                                                       # 4. run everything
```

**The starting point is `run_study.sh`.** It runs every analysis step in order and finishes with the
tables, figures and numbers of the manuscript. Each step is a separate script in `scripts/`, so you can
also run or inspect them one at a time.

---

## Step by step

### 1. Install
Python 3.8 is required; the published results were produced with the exact package versions in
`requirements-lock.txt`.

```bash
conda env create -f environment.yml && conda activate hnc-popgraph
# without conda:
pip install numpy==1.24.3 "setuptools>=65" && pip install -r requirements.txt
```
A GPU is not needed: the networks are small and are trained on the CPU.

### 2. Download and arrange the data
No patient data are included in this repository. Both collections are public on The Cancer Imaging
Archive (TCIA). Download the images with the NBIA Data Retriever.

| Cohort | Download from TCIA |
|---|---|
| Head-Neck-PET-CT | <https://doi.org/10.7937/K9/TCIA.2017.8oje5q00>: the DICOM collection, `INFOclinical_HN_Version2_30may2018.xlsx` (outcomes) and `INFO_GTVcontours_HN.xlsx` (names of the GTV structures). PET is not used. |
| HN1 (MAASTRO) | <https://doi.org/10.7937/tcia.2019.8kap372n>: the DICOM collection (CT and RTSTRUCT) and the clinical-data CSV |

Then let `scripts/organize_data.py` build the folder layout. It creates links, not copies (add
`--copy` to copy), and never modifies the downloads.

```bash
python scripts/organize_data.py hnpet --src <download>/Head-Neck-PET-CT \
       --clinical <download>/INFOclinical_HN_Version2_30may2018.xlsx \
       --gtv-list <download>/INFO_GTVcontours_HN.xlsx --out data/Head-Neck-PET-CT
python scripts/organize_data.py hn1 --src <download>/HEAD-NECK-RADIOMICS-HN1 \
       --clinical "<download>/<HN1 clinical data>.csv" --out data/MAASTRO
```

Result:
```
data/
├── Head-Neck-PET-CT/
│   ├── Images/HN-<centre>-<nnn>/CT/*.dcm, RTSTRUCT/*.dcm   CT of the PET/CT and the GTV structure set
│   ├── INFOclinical_HN_Version2_30may2018.xlsx             clinical data and outcomes
│   ├── INFO_GTVcontours_HN.xlsx                            primary and nodal GTV structure names
│   └── gtv_primary_rois.csv                                primary-GTV structure(s) per patient (created)
└── MAASTRO/
    ├── Images/<HN1xxxx>/CT/*.dcm, RTSTRUCT/*.dcm
    └── clinical_data.csv
```
The data can live elsewhere: set `HNC_DATA_DIR` (or `HNC_HNPET_DIR` and `HNC_MAASTRO_DIR`). Results go
to `outputs/` unless `HNC_OUTPUT_DIR` is set.

### 3. Check
```bash
python scripts/check_data.py
```
This checks the packages, the files and the patient and event counts against the paper. A correct
setup ends with:
```
Patients with outcome data (events), compared with the paper
  [OK] HNPET    LR: 276 patients (30 events); paper 276 (30)
  [OK] HNPET    DM: 276 patients (27 events); paper 276 (27)
  [OK] HNPET    OS: 164 patients (44 events); paper 164 (44)
  [OK] MAASTRO  LR: 121 patients (30 events); paper 121 (30)
  [OK] MAASTRO  DM: 109 patients (7 events); paper 109 (7)
  [OK] MAASTRO  OS: 135 patients (45 events); paper 135 (45)

All checks passed: run ./run_study.sh
```

### 4. Run
```bash
./run_study.sh               # all steps
N_JOBS=12 ./run_study.sh     # more parallel workers (default 4)
./run_study.sh 6             # resume from step 6, reusing the outputs of steps 1-5
```
Each step prints its progress, and a copy is kept in `outputs/logs/stepNN.log`.

| Step | Script | What it does | Main output (in `outputs/`) |
|---|---|---|---|
| 1 | `preprocess.py` | resamples CT and primary GTV to 1 mm and crops them | `preprocessed/<patient>_preprocessed.npz` |
| 2 | `extract_primary_gtv_features.py` | 1223 radiomic features of the primary GTV | `features/primary_gtv_features.csv` |
| 3 | `tumour_burden.py` | primary, nodal and total GTV volumes, number of nodal GTVs | `features/tumour_burden.csv` |
| 4 | `extract_whole_tumour_features.py` | 479 radiomic features of the whole tumour (primary + nodal GTVs) | `features/whole_tumour_features.csv` |
| 5 | `cohort_summary.py` | patients and events per endpoint and cohort | `results/cohort_summary.csv` |
| 6 | `train_population_graph.py --region total --clinical burden` | **main analysis**: the full grid of graph networks and MLPs, for 3 endpoints × 2 designs; creates the splits | `splits/*.json`, `results/population_graph/popgnn_burden_total_*` |
| 7 | `train_population_graph.py` (3 more runs) | the other three representations of the 2×2 ablation | `results/population_graph/popgnn_total_*`, `popgnn_burden_*`, `popgnn_*` |
| 8 | `compare_graph_mlp.py` | best graph vs best MLP vs the CNN; paired bootstrap; matched comparison | `results/population_graph/*_family_selection.csv`, `*_graph_vs_mlp_paired.csv`, `*_graph_vs_mlp_matched.csv` |
| 9 | `seed_robustness.py` | retrains every chosen model with 10 new 10-seed ensembles | `results/population_graph/popgnn_burden_total_seed_robustness.csv` |
| 10 | `train_conventional_classifiers.py` | logistic regression and random forests on the same features and splits | `results/conventional/classical_total_*` |
| 11 | `manuscript/scripts/make_assets.py` | all tables, figures and quoted numbers of the manuscript | `manuscript/generated/` |

**Run time.** Steps 1–4 read every DICOM series once, so their duration depends mainly on disk speed.
Steps 6 and 7 dominate the total. One network training takes 0.2–1.1 s on one CPU core, and one
representation needs about 67,000 trainings (18 splits × 370 configurations × 10 seeds). That is
roughly 10 CPU-hours per representation: about 3 hours with `N_JOBS=4`, or under an hour with
`N_JOBS=18` or more (there are 18 splits to run in parallel). Steps 6 and 7 together are four
representations. Step 9 needs about 1.5 CPU-hours; steps 5, 8 and 11 take minutes.

---

## What to expect

**During step 6** the script reports, for every endpoint, design and input set, the configuration
chosen on validation and its external result. For distant metastasis:
```
DM MATEUS_CV     imaging          chosen orig+shape|ucorr|4|g5|mlp|imaging             val 0.78 test 0.88 fold-mean 0.85
DM MATEUS_CV     imaging+clinical chosen orig+shape|ucorr|4|g5|mlp|imaging+clinical    val 0.80 test 0.87 fold-mean 0.86
DM MATEUS_COHORT imaging          chosen full|lasso|8|g20|sage|imaging                 val 0.90 test 0.87 fold-mean 0.87
DM MATEUS_COHORT imaging+clinical chosen full|lasso|8|g20|gcn|imaging+clinical         val 0.89 test 0.85 fold-mean 0.85
```
**Step 8** prints the central comparison. The table gives the external AUC of the model chosen on
validation among all models (`all`), graph networks only (`graph`) and MLPs only (`mlp`), next to the
published CNN (`mateus`). CV rows are means of the five per-fold AUCs:
```
family                                 all  graph    mlp  mateus
task design        inputs
DM   MATEUS_COHORT imaging           0.870  0.870  0.861    0.89
     MATEUS_CV     imaging           0.850  0.848  0.850    0.83
LR   MATEUS_COHORT imaging           0.627  0.627  0.636    0.45
OS   MATEUS_CV     imaging           0.734  0.734  0.742    0.63
...
graph  beats Mateus in 9/12
mlp    beats Mateus in 8/12
```
It then lists the paired bootstrap of the chosen graph network against the chosen MLP on MAASTRO
(difference in AUC, 95% interval and p value) for all 18 endpoint × design × input combinations.

**At the end** `manuscript/generated/` holds every table (`tables/*.tex`), every figure
(`figures/*.pdf`, `*.png`) and `tables/numbers.tex`, the file that supplies every number quoted in the
manuscript text. With the published environment these files are identical to those in the repository.

---

## Reading the results

The summary tables from our own run are included in [`results/`](results/) (no patient-level data), so every number in the manuscript can be checked without re-running the study.

Each grid run writes two files per endpoint and design to `outputs/results/population_graph/`:

* `<prefix>_<task>_<design>.csv`: one row per configuration, with `val_auc` (used for choosing),
  `test_auc` (MAASTRO, pooled over folds), `test_auc_fold_mean`, `_fold_min` and `_fold_max`;
* `<prefix>_pred_<task>_<design>.csv.gz`: the validation and test probabilities of every
  configuration and patient.

A configuration is written as `featureset|selector|k|g<neighbours>|model|inputs`. For example,
`full|lasso|8|g20|sage|imaging` means all features, LASSO selection, 8 features, 20 training neighbours
per node, GraphSAGE, imaging only. The prefix names the representation:

| Prefix | Radiomic region | Clinical variables |
|---|---|---|
| `popgnn_burden_total` | whole tumour | tumour burden (main analysis) |
| `popgnn_total` | whole tumour | volume bins (as in the CNN study) |
| `popgnn_burden` | primary GTV | tumour burden |
| `popgnn` | primary GTV | volume bins |

Where the results appear in the manuscript:

| Manuscript | Source |
|---|---|
| Table 1 (cohorts) | `results/cohort_summary.csv`, `features/tumour_burden.csv`, clinical data |
| Table 3, Fig. 3 (external AUC vs the CNN) | `popgnn_burden_total_family_selection.csv`, `results/conventional/` |
| Table 4, Fig. 4 (ablation) | the four `*_family_selection.csv` files |
| Table 5, Fig. 5 (graph vs MLP) | `popgnn_burden_total_graph_vs_mlp_paired.csv`, `_matched.csv` |
| Fig. 6, Table S2 (seeds) | `popgnn_burden_total_seed_robustness.csv` |

---

## Repository layout

```
├── run_study.sh                  START HERE: runs steps 1-11 in order
├── scripts/
│   ├── organize_data.py          arrange the TCIA downloads into data/
│   ├── check_data.py             check packages, files and patient counts
│   ├── preprocess.py             step 1
│   ├── extract_primary_gtv_features.py        step 2
│   ├── tumour_burden.py          step 3
│   ├── extract_whole_tumour_features.py       step 4
│   ├── cohort_summary.py         step 5
│   ├── train_population_graph.py              steps 6-7 (graph networks and MLP)
│   ├── compare_graph_mlp.py      step 8
│   ├── seed_robustness.py        step 9
│   └── train_conventional_classifiers.py      step 10
├── hnc_popgraph/                 shared code used by the scripts
│   ├── config.py                 cohorts, endpoints, designs, radiomics settings, paths
│   ├── data/
│   │   ├── loader.py             DICOM CT and RTSTRUCT loading, contour rasterisation
│   │   ├── clinical.py           outcome labels and clinical variables
│   │   ├── delineations.py       primary and nodal GTV structures, whole-tumour mask
│   │   ├── preprocessing.py      1 mm resampling and cropping
│   │   └── splits.py             eligible patients and the study splits
│   ├── features.py               feature tables, feature subsets, node-feature preprocessing
│   ├── selection.py              mRMR and LASSO
│   ├── stats.py                  AUC and bootstrap confidence intervals
│   └── benchmark.py              published CNN results (Mateus et al. 2023)
├── manuscript/                   the paper: main.tex, supplementary.tex, title page, cover letter,
│   │                             figures, tables, submission figures, Overleaf bundle
│   ├── scripts/                  make_assets.py (step 11), make_checklists.py, make_report.py
│   └── generated/                raw output of make_assets.py
├── results/                    summary tables from our run (aggregated, no patient data)
├── environment.yml, requirements.txt, requirements-lock.txt
├── data/                         TCIA data (you create it; not in the repository)
└── outputs/                      everything the pipeline produces (not in the repository)
```

---

## Reproducibility

* **No information from the test cohort is used for training or model choice.** Feature
  preprocessing, feature selection and the clinical encoder are fitted on the training patients of
  each split. Validation and test patients only receive edges from training patients, so their labels
  cannot reach other nodes.
* The splits are deterministic (seed 42) and saved in `outputs/splits/`. Every network is trained with
  fixed seeds on one CPU thread.
* With the published environment, retraining the reported models from the extracted features
  reproduces the saved predictions to within 0.00002 (floating-point noise), the conventional
  classifiers reproduce exactly, and every manuscript table and number is regenerated identically.
* Reporting follows TRIPOD+AI, CLAIM 2024 and CLEAR; the completed checklists, with METRICS and RQS
  scores, are in the supplementary information (`manuscript/supplementary.pdf`).

## Licence
MIT (see `LICENSE`). The TCIA data are subject to the TCIA data usage policies.
