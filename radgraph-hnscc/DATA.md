# Data: download and arrange

Both cohorts are public. **No patient data are distributed with this repository.** After the steps
below, `python check_setup.py` compares your patient and event counts with those of the paper.

## 1. Download

| Cohort | Role | What to download |
|---|---|---|
| **Head-Neck-PET-CT** (CHUM, CHUS, HGJ, HMR; Québec, Canada) | training / validation | (a) **Images:** the HECKTOR challenge training data (NIfTI CT `<ID>__CT.nii.gz` + segmentation `<ID>.nii.gz`, label 1 = primary GTV, 2 = nodal GTV), from <https://hecktor.grand-challenge.org/>. (b) **Outcomes:** `INFOclinical_HN_Version2_30may2018.xlsx` from the TCIA collection page, <https://doi.org/10.7937/K9/TCIA.2017.8oje5q00>. |
| **HN1** (MAASTRO Clinic, the Netherlands) | external test only | DICOM CT + RTSTRUCT and the clinical-data CSV from the TCIA collection page, <https://doi.org/10.7937/tcia.2019.8kap372n>. Download the images with the NBIA Data Retriever. |

Only the four Head-Neck-PET-CT centres are used. HECKTOR patients from other centres are ignored.

## 2. Arrange

The pipeline expects this layout:

```
<data dir>/                                   default: radgraph-hnscc/data  (or set RADGRAPH_DATA_DIR)
├── HNPET/                                    "HECKTOR 2026 Training Data/" is also accepted
│   ├── INFOclinical_HN_Version2_30may2018.xlsx      sheets CHUM, CHUS, HGJ, HMR
│   └── Images/<ID>/<ID>__CT.nii.gz, <ID>.nii.gz      e.g. Images/CHUM-001/CHUM-001__CT.nii.gz
└── MAASTRO/
    ├── clinical_data.csv                     HN1 clinical table (columns patient_id, locoregional_recurrence, …)
    └── Images/<HN1xxxx>/CT/*.dcm, RTSTRUCT/*.dcm
```

`tools/organize_data.py` builds this layout from the downloads. By default it creates symlinks, so it
uses no extra disk space; add `--copy` to copy the files instead. The downloaded folders are not modified.

```bash
cd radgraph-hnscc
python tools/organize_data.py hecktor --src /downloads/hecktor \
       --clinical /downloads/INFOclinical_HN_Version2_30may2018.xlsx --out data/HNPET
python tools/organize_data.py hn1 --src /downloads/NBIA/HEAD-NECK-RADIOMICS-HN1 \
       --clinical "/downloads/<HN1 clinical data>.csv" --out data/MAASTRO
```

The `hn1` command reads `PatientID` and `Modality` from each DICOM header, so any NBIA folder nesting
works.

## 3. Point the code to the data (optional)

You don't need to edit any code. Set environment variables instead:

| Variable | Meaning | Default |
|---|---|---|
| `RADGRAPH_DATA_DIR` | folder holding `HNPET/` and `MAASTRO/` | `radgraph-hnscc/data` |
| `RADGRAPH_HNPET_DIR`, `RADGRAPH_MAASTRO_DIR` | the two cohort folders individually | inside the data dir |
| `RADGRAPH_OUTPUT_DIR` | caches, runs, results, final models | `radgraph-hnscc/outputs` |
| `RADGRAPH_LOGS_DIR` | log files | `radgraph-hnscc/logs` |
| `RADGRAPH_PREPROCESS` | preprocessing profile `v3` (paper) or `v4` | `v4` (`run_mateus_study.sh`: `v3`) |

## 4. Check

```bash
python check_setup.py
```

Expected counts, which are those of the paper:

| | Patients | LR, patients (events) | DM, patients (events) | OS, patients (events) |
|---|:-:|:-:|:-:|:-:|
| HNPET | 199 | 190 (13) | 191 (12) | 92 (19) |
| MAASTRO | 137 | 121 (30) | 109 (7) | 135 (45) |

Different counts don't stop the pipeline, but your results will then differ from the paper. The most
likely cause is a different image set for Head-Neck-PET-CT. The study used 199 of the 298 patients in the
clinical file: those with a CT and primary GTV in the HECKTOR release.
