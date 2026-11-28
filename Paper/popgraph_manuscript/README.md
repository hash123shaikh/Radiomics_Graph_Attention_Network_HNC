# Population-graph manuscript (Arabian Journal for Science and Engineering submission)

*Graph neural networks on primary and nodal tumour radiomics for outcome prediction in head and neck
cancer: a multicentre external validation study.*

| Path | Content |
|---|---|
| `main.tex` | manuscript (Springer Nature `sn-jnl`, two-column) |
| `supplementary.tex` | Online Resource 1: Tables S1–S2 and the TRIPOD+AI, CLAIM 2024, CLEAR, METRICS and RQS checklists (Tables S3–S7) |
| `title_page.tex`, `cover_letter.tex` | separate submission files |
| `fig2_standalone.tex` | builds Fig. 2 (TikZ source `figures/Fig2.tex`) as a separate PDF |
| `figures/`, `tables/` | figures and tables used by the manuscript (edited for the journal; these are the authoritative versions) |
| `submission_figures/` | Fig1–Fig6 as vector PDF and 600 dpi TIFF for upload |
| `dropped_figures/` | figures removed from the manuscript (patient-graph example), kept for reference |
| `GAT_overleaf.zip` | Overleaf bundle of all sources |
| `scripts/make_assets.py` | regenerates figures, tables and `numbers.tex` from `grafs-gat/outputs/` into `generated/` |
| `scripts/make_checklists.py` | regenerates the checklist tables into `generated/tables/` |
| `scripts/make_report.py` | writes `report/popgraph_report.{docx,pdf}` and `grafs-gat/popgraph/REPORT.md` |
| `generated/` | raw script output; never overwrites the manuscript's own `figures/` and `tables/` |
| `report/` | study report |

## Compile
Overleaf: upload `GAT_overleaf.zip`; `main.tex` is the main document (pdfLaTeX → BibTeX → pdfLaTeX ×2).
Locally: `tectonic main.tex`, and likewise for `supplementary.tex`, `title_page.tex`, `cover_letter.tex`.

## After a re-run of the analysis
```bash
cd grafs-gat
POPGNN_CLIN=burden POPGNN_FEAT=total python popgraph/run_popgnn.py
python popgraph/graph_vs_mlp.py --prefix popgnn_burden_total
POPGNN_CLIN=burden POPGNN_FEAT=total python popgraph/seed_robustness.py
cd gtv_only && CLS_FEAT=total python run_classical.py
cd ../.. && python Paper/popgraph_manuscript/scripts/make_assets.py
```
Then copy `generated/tables/numbers.tex` to `tables/numbers.tex` (every number in the text updates), and
compare the other files in `generated/` with the edited versions in `figures/` and `tables/`.
