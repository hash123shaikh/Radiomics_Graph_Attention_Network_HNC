#!/usr/bin/env bash
# Runs the whole study, from the arranged data to every table, figure and number of the manuscript.
#
#   ./run_study.sh            all steps
#   ./run_study.sh 6          start at step 6 (earlier outputs are reused)
#   N_JOBS=8 ./run_study.sh   parallel workers (default 4)
#
# Before the first run: arrange the data (scripts/organize_data.py) and run scripts/check_data.py.
# Progress of each step is printed and saved in outputs/logs/.
set -euo pipefail
cd "$(dirname "$0")"
FROM=${1:-1}
J=${N_JOBS:-4}
PY=${PYTHON:-python}
mkdir -p outputs/logs

step() {                       # step <number> <title> <command...>
    local n=$1 title=$2; shift 2
    if [ "$n" -lt "$FROM" ]; then return; fi
    echo "=== Step $n: $title  ($(date '+%H:%M'))"
    "$@" 2>&1 | tee "outputs/logs/step$(printf %02d "$n").log"
}

step 1  "preprocess CT and primary GTV"         $PY scripts/preprocess.py --n_jobs "$J"
step 2  "primary-GTV radiomic features"         $PY scripts/extract_primary_gtv_features.py --n_jobs "$J"
step 3  "tumour burden (GTV volumes)"           $PY scripts/tumour_burden.py --n_jobs "$J"
step 4  "whole-tumour radiomic features"        $PY scripts/extract_whole_tumour_features.py --n_jobs "$J"
step 5  "cohort summary"                        $PY scripts/cohort_summary.py
step 6  "graph networks: whole tumour + burden (main analysis)" \
        $PY scripts/train_population_graph.py --region total --clinical burden --n_jobs "$J"
step 7  "graph networks: ablation, 3 other representations" bash -c "
        $PY scripts/train_population_graph.py --region total   --clinical mateus --n_jobs $J &&
        $PY scripts/train_population_graph.py --region primary --clinical burden --n_jobs $J &&
        $PY scripts/train_population_graph.py --region primary --clinical mateus --n_jobs $J"
step 8  "graph versus edgeless network (all four representations)" bash -c "
        for p in popgnn_burden_total popgnn_total popgnn_burden popgnn; do $PY scripts/compare_graph_mlp.py --prefix \$p; done"
step 9  "seed robustness"                       $PY scripts/seed_robustness.py --n_jobs "$J"
step 10 "conventional classifiers"              $PY scripts/train_conventional_classifiers.py --region total --n_jobs "$J"
step 11 "manuscript tables, figures, numbers"   $PY manuscript/scripts/make_assets.py
echo "=== Done. Results: outputs/results/   Manuscript assets: manuscript/generated/"
