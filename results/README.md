# Results snapshot

The summary tables behind every number, table and figure in the manuscript, so that results can be checked without re-running the study. They are copies of the files that `run_study.sh` writes to `outputs/results/`.

All tables are aggregated: one row per model configuration or per cohort. They contain **no patient identifiers, no per-patient predictions and no image-derived features**. Per-patient predictions, features and preprocessed images stay out of the repository because they are derived from patient data; run the pipeline on the public TCIA data to regenerate them.

## Files

| File | Contents | Used for |
|---|---|---|
| `cohort_summary.csv` | Patients, events and event rate per endpoint and cohort | Table 1 |
| `population_graph/<prefix>_<task>_<design>.csv` | Every graph and MLP configuration: validation AUC, external test AUC, fold mean, min and max, test patients and events | All graph-network results |
| `population_graph/<prefix>_family_selection.csv` | Configuration chosen by validation AUC for each model family, with bootstrap 95% CI and comparison with the published CNN | Tables 3 and 4, Figs 3 and 4 |
| `population_graph/popgnn_burden_total_graph_vs_mlp_matched.csv` | Graph minus MLP AUC over matched configurations | Table 5, Fig 5 |
| `population_graph/popgnn_burden_total_graph_vs_mlp_paired.csv` | Chosen graph versus chosen MLP on the external test set, paired bootstrap | Table 5 |
| `population_graph/popgnn_burden_total_seed_robustness.csv` | Repeated training with new seeds for the chosen configurations | Fig 6, Table S2 |
| `conventional/classical[_total]_<task>_<design>.csv` | Conventional classifiers on the same features and splits | Supplementary comparison |

## Prefixes

| Prefix | Tumour definition | Clinical variables |
|---|---|---|
| `popgnn_burden_total` | Whole tumour (primary and nodal GTVs) | Nodal burden (main analysis) |
| `popgnn_total` | Whole tumour | As in Mateus et al. |
| `popgnn_burden` | Primary GTV only | Nodal burden |
| `popgnn` | Primary GTV only | As in Mateus et al. |

Configuration strings read `featureset|selector|k|g<neighbours>|model|inputs`; see the main README for details.
