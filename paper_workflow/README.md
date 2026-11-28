# paper_workflow — project documents (November 2026)

Rewritten versions of the original May 2026 documents, which are kept unchanged in `archive_v1/`. Each is a Word file
with a PDF copy, and each describes the **current** code and the **current** study (complete 298-patient
Head-Neck-PET-CT collection, preprocessing version 4; updated 18 November 2026).

| New document | Replaces | Content |
|---|---|---|
| `01_Study_Rationale_Design_and_Methods_v2` | `GNN_Paper -- Introduction_and_Methods` | background, aims, data, endpoints, Mateus et al. validation design, every pipeline stage, statistics, findings to date |
| `02_Graph_Attention_Networks_Explained_v2` | `GAT_Questions_and_Answers` | plain-English GAT concepts, the exact network of this study (with parameter count), training, 16 questions and answers, glossary |
| `03_Implementation_History_Audit_and_Roadmap_v2` | original bug log (`archive_v1/`) | the 13 engineering bugs, the methodological audit (what version 1 got wrong and how it is fixed), open issues, prioritised roadmap, lessons |
| `04_Stage2_Supervoxel_Generation_v2` | `step2_slic_supervoxel_documentation` | SLIC-zero step by step, body mask and soft-tissue window (version 4), version 3 vs 4 statistics on all analysed patients, open improvements |

Regenerate (figures need the project environment; documents use system Python with python-docx):

```bash
cd scripts
python make_figures.py         # project environment (conda activate GNNvenv)
python3 doc1_methods.py; python3 doc2_gat.py; python3 doc3_audit.py; python3 doc4_slic.py   # needs python-docx
```
