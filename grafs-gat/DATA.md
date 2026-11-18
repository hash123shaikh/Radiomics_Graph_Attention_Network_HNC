# Data: download and arrange

Both cohorts are public. **No patient data are distributed with this repository.** After the steps
below, `python check_setup.py` compares your patient and event counts with those of the paper.

## 1. Download

| Cohort | Role | What to download |
|---|---|---|
| **Head-Neck-PET-CT** (CHUM, CHUS, HGJ, HMR; Québec, Canada; 298 patients) | training / validation | From the TCIA collection page, <https://doi.org/10.7937/K9/TCIA.2017.8oje5q00>: (a) **Images:** the full DICOM collection, downloaded with the NBIA Data Retriever; (b) **Outcomes:** `INFOclinical_HN_Version2_30may2018.xlsx`; (c) **GTV names:** `INFO_GTVcontours_HN.xlsx`. Only CT and the primary-GTV RTSTRUCT are used; PET is not needed. |
| **HN1** (MAASTRO Clinic, the Netherlands) | external test only | DICOM CT + RTSTRUCT and the clinical-data CSV from the TCIA collection page, <https://doi.org/10.7937/tcia.2019.8kap372n>. Download the images with the NBIA Data Retriever. |


## 2. Arrange

The pipeline expects this layout:

```
<data dir>/                                   default: grafs-gat/data  (or set GRAFS_DATA_DIR)
├── Head-Neck-PET-CT/                         (cohort name in the code: HNPET)
│   ├── INFOclinical_HN_Version2_30may2018.xlsx      sheets CHUM, CHUS, HGJ, HMR
│   ├── INFO_GTVcontours_HN.xlsx                     primary/nodal GTV ROI names (TCIA)
│   ├── gtv_primary_rois.csv                         primary GTV ROI name(s) per patient (made by organize_data.py)
│   └── Images/<HN-CHUM-001 …>/CT/*.dcm, RTSTRUCT/*.dcm    same layout as MAASTRO
└── MAASTRO/
    ├── clinical_data.csv                     HN1 clinical table (columns patient_id, locoregional_recurrence, …)
    └── Images/<HN1xxxx>/CT/*.dcm, RTSTRUCT/*.dcm
```

`tools/organize_data.py` builds this layout from the downloads. By default it creates symlinks, so it
uses no extra disk space; add `--copy` to copy the files instead. The downloaded folders are not modified.

```bash
cd grafs-gat
python tools/organize_data.py hnpet --src /downloads/NBIA/Head-Neck-PET-CT \
       --clinical /downloads/INFOclinical_HN_Version2_30may2018.xlsx \
       --gtv-list /downloads/INFO_GTVcontours_HN.xlsx --out data/Head-Neck-PET-CT
python tools/organize_data.py hn1 --src /downloads/NBIA/HEAD-NECK-RADIOMICS-HN1 \
       --clinical "/downloads/<HN1 clinical data>.csv" --out data/MAASTRO
```

The `hn1` command reads `PatientID` and `Modality` from each DICOM header, so any NBIA folder nesting
works.

**Which Head-Neck-PET-CT series are used.** A patient has 6–11 series (PET/CT, planning CT, dose,
plans, several structure sets). The `hnpet` command keeps, per patient:

- **CT** = the CT of the diagnostic PET/CT (the CT series in the same frame of reference as the
  structure set below; the TomoTherapy planning kVCT is never used);
- **RTSTRUCT** = `RTstructCTsimCTPETCT` (curated GTV on that CT; 207 patients) or, when absent,
  `RTstructCTsimPETPETCT` (GTV on the PET of the same PET/CT, same frame of reference; 91 patients,
  mostly CHUS). The loader rasterises contours from their physical coordinates, so both work.
- **Primary GTV** = the ROI name(s) given in `INFO_GTVcontours_HN.xlsx` (sometimes several ROIs, whose
  union is used, e.g. `GTVp G;GTVp D`), written to `gtv_primary_rois.csv` with the case found in the file.

Two CTs (HN-CHUM-001, HN-CHUM-002) miss one slice in the TCIA release; the loader rebuilds them on a
uniform grid. One structure set (HN-HGJ-003) skips one contour plane; the loader fills it from the
nearest plane.

## 3. Point the code to the data (optional)

You don't need to edit any code. Set environment variables instead:

| Variable | Meaning | Default |
|---|---|---|
| `GRAFS_DATA_DIR` | folder holding `HNPET/` and `MAASTRO/` | `grafs-gat/data` |
| `GRAFS_HNPET_DIR`, `GRAFS_MAASTRO_DIR` | the two cohort folders individually | `Head-Neck-PET-CT/` (else `HNPET/`, `HECKTOR 2026 Training Data/`) and `MAASTRO/` inside the data dir |
| `GRAFS_OUTPUT_DIR` | caches, runs, results, final models | `grafs-gat/outputs` |
| `GRAFS_LOGS_DIR` | log files | `grafs-gat/logs` |
| `GRAFS_PREPROCESS` | preprocessing profile `v4` (paper) or `v3` (release v0.1.0-models) | `v4` |

## 4. Check

```bash
python check_setup.py
```

Expected counts, which are those of the paper:

| | Patients | LR, patients (events) | DM, patients (events) | OS, patients (events) |
|---|:-:|:-:|:-:|:-:|
| HNPET (Head-Neck-PET-CT) | 294 | 276 (30) | 276 (27) | 164 (44) |
| MAASTRO | 137 | 121 (30) | 109 (7) | 135 (45) |

Different counts don't stop the pipeline, but your results will then differ from the paper. The most
likely cause is a different image set for Head-Neck-PET-CT: the study uses all 298 patients of the TCIA
collection (294 with a valid label for at least one endpoint). The earlier HECKTOR NIfTI release holds
only 201 of them (oropharynx only) and gives 199 patients with 13 LR, 12 DM and 19 OS events.
