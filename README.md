# 1_Idea — radiomic graph attention networks for head and neck cancer

QIRAIL Lab, Department of Radiation Oncology, CMC Vellore.

| Folder / file | Content |
|---|---|
| `radgraph-hnscc/` | The study code (Mateus et al. design: 5-fold CV and cohort split on Head-Neck-PET-CT, external test on MAASTRO; LR, DM and OS), data, trained models and results. **Start with `radgraph-hnscc/README.md`.** |
| `radgraph-hnscc/data/` | Images and clinical files: `HECKTOR 2026 Training Data/` (Head-Neck-PET-CT, HECKTOR release: CT, PET and segmentations of 201 oropharyngeal patients + clinical spreadsheet) and `MAASTRO/` (HN1: CT + RTSTRUCT + clinical CSV). |
| `radgraph-hnscc/outputs/` | `cache/` (preprocessed CT and features), `runs/` (every model, graph and per-fold result), `results/` (study CSVs and figures), `final_models/` (chosen models for `predict.py`). |
| `Paper/manuscript/` | The manuscript (`main.tex`), Supplementary Information (`supplementary.tex`), references, figures and tables. `scripts/make_paper_assets.py` regenerates every figure, table and number from `radgraph-hnscc/outputs/results/`. |
| `Paper/RadGraph-FS_manuscript.zip` | The manuscript folder packed for upload to Overleaf. |
| `Paper/*.pdf, *.pptx, *.docx` | Reference papers (RadGraph, Graph-FS, Mateus et al.), earlier abstract, poster and slides. |
| `architecture_diagram/` | Architecture figures; `Radiomics GRaph Attention Network Architecture.png` is the one used in the manuscript (Figure 1). |
| `paper_workflow/` | Earlier drafts and notes (introduction/methods draft, GAT Q&A, bug log, SLIC documentation). |
| `CodeRelease/` | Original RadGraph authors' code (Bae et al.), kept as reference. |

## Typical commands

```bash
cd radgraph-hnscc
./run_mateus_study.sh                       # full study (all endpoints, both designs, both checkpoint rules)
python export_results.py                    # collect results -> outputs/results/
python predict.py --task LR --dataset MAASTRO   # inference with the saved models
/home/radiomicsserver/anaconda3/envs/GNNvenv/bin/python ../Paper/manuscript/scripts/make_paper_assets.py   # refresh the paper
```
