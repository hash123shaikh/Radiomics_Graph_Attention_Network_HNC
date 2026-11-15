# paper_workflow_v2 — upgraded project documents (November 2026)

Rewritten versions of the documents in `../paper_workflow/` (originals kept unchanged). Each is a Word file
with a PDF copy, and each describes the **current** code and the **current** study.

| New document | Replaces | Content |
|---|---|---|
| `01_Study_Rationale_Design_and_Methods_v2` | `GNN_Paper -- Introduction_and_Methods` | background, aims, data, endpoints, Mateus et al. validation design, every pipeline stage, statistics, findings to date |
| `02_Graph_Attention_Networks_Explained_v2` | `GAT_Questions_and_Answers` | plain-English GAT concepts, the exact network of this study (with parameter count), training, 16 questions and answers, glossary |
| `03_Implementation_History_Audit_and_Roadmap_v2` | `RadGraph_Bug_Log_and_Future_Work` | the 13 engineering bugs, the methodological audit (what version 1 got wrong and how it is fixed), open issues, prioritised roadmap, lessons |
| `04_Stage2_Supervoxel_Generation_v2` | `step2_slic_supervoxel_documentation` | SLIC-zero step by step, observed behaviour on all 336 patients, two corrections and recommended improvements |

Regenerate (figures need the project environment; documents use system Python with python-docx):

```bash
cd scripts
python make_figures.py              # in the project environment (conda activate GNNvenv)
python3 doc1_methods.py; python3 doc2_gat.py; python3 doc3_audit.py; python3 doc4_slic.py   # needs python-docx
```
