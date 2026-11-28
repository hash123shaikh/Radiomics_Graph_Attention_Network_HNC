# GRAFS-GAT manuscript (Springer Nature LaTeX template)

| File | Content |
|---|---|
| `main.tex` | the manuscript (compact A4 article layout, numbered references; `sn-jnl.cls` kept for a journal-formatted version) |
| `supplementary.tex` | Supplementary Information (separate PDF) |
| `references.bib` | bibliography; check every entry against the publisher record before submission |
| `figures/` | all figures (generated, see below) |
| `tables/` | all tables + `numbers.tex` (every number quoted in the text, as LaTeX macros) |
| `scripts/make_paper_assets.py` | regenerates `figures/` and `tables/` from `grafs-gat/outputs/results/` (current results: full 298-patient Head-Neck-PET-CT, preprocessing v4) |
| `sn-jnl.cls`, `bst/` | the official Springer Nature template files (for a journal-formatted version) |

## Compile on Overleaf
1. Overleaf → **New Project → Upload Project** → choose `GRAFS-GAT_manuscript.zip`.
2. `main.tex` compiles as the main document (pdfLaTeX → BibTeX → pdfLaTeX ×2, which Overleaf does automatically).
3. For the supplement: Menu → *Main document* → `supplementary.tex` → Recompile (then switch back).

## Compile locally
```bash
latexmk -pdf main.tex
latexmk -pdf supplementary.tex
```

## Updating after a re-run
```bash
cd ../../grafs-gat && ./run_mateus_study.sh          # re-train + export_results.py
python ../Paper/grafs-gat_manuscript/scripts/make_paper_assets.py   # set GRAFS_OUTPUT_DIR if the results live elsewhere
```
The tables, figures and every number in the text update automatically. Re-read the Results/Discussion
wording afterwards, because the direction of some findings may change, and delete the orange draft notes
(`\draftfalse` in `main.tex`).

## Before submission (to do)
- Authors, affiliations, funding, CRediT statement (placeholders in `main.tex`).
- Choose the target journal and switch the reference style if needed
  (`sn-nature` for Scientific Reports / Nature Portfolio).
- Verify the references in `references.bib`.
