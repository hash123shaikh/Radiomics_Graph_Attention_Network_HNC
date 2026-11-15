# RadGraph-FS manuscript (Springer Nature LaTeX template)

| File | Content |
|---|---|
| `main.tex` | the manuscript (compact A4 article layout, numbered references; `sn-jnl.cls` kept for a journal-formatted version) |
| `supplementary.tex` | Supplementary Information (separate PDF) |
| `references.bib` | bibliography; check every entry against the publisher record before submission |
| `figures/` | all figures (generated, see below) |
| `tables/` | all tables + `numbers.tex` (every number quoted in the text, as LaTeX macros) |
| `scripts/make_paper_assets.py` | regenerates `figures/` and `tables/` from `radgraph-hnscc/outputs/results/` |
| `sn-jnl.cls`, `bst/` | the official Springer Nature template files (for a journal-formatted version) |

## Compile on Overleaf
1. Overleaf → **New Project → Upload Project** → choose `RadGraph-FS_manuscript.zip`.
2. `main.tex` compiles as the main document (pdfLaTeX → BibTeX → pdfLaTeX ×2, which Overleaf does automatically).
3. For the supplement: Menu → *Main document* → `supplementary.tex` → Recompile (then switch back).

## Compile locally
```bash
latexmk -pdf main.tex
latexmk -pdf supplementary.tex
```

## Updating after a re-run (e.g. the full Head-Neck-PET-CT image set)
```bash
cd ../../radgraph-hnscc && ./run_mateus_study.sh          # re-train + export_results.py
python ../Paper/manuscript/scripts/make_paper_assets.py   # set RADGRAPH_OUTPUT_DIR if the results live elsewhere
```
The tables, figures and every number in the text update automatically. Re-read the Results/Discussion
wording afterwards, because the direction of some findings may change, and delete the orange draft notes
(`\draftfalse` in `main.tex`).

## Before submission (to do)
- Authors, affiliations, funding, CRediT statement, repository URL (placeholders in `main.tex`).
- Confirm the source of the Canadian images (HECKTOR subset vs. TCIA planning CT; see the draft note).
- Choose the target journal and switch the reference style if needed
  (`sn-nature` for Scientific Reports / Nature Portfolio).
- Verify the references in `references.bib`.
