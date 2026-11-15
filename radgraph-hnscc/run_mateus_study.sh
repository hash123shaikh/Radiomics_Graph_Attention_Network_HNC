#!/usr/bin/env bash
# Full study, Mateus et al. (Sci Rep 2023) design: HNPET (4 Canadian centres) -> external MAASTRO.
#   1. preprocess + extract every eligible patient (cached)
#   2. LR, DM, OS in parallel: 5-fold CV and cohort split, checkpoint = lowest validation loss (primary)
#   3. same with Mateus' train-val AUC-gap checkpoint rule (sensitivity analysis)
#   4. export: $RADGRAPH_OUTPUT_DIR/results/ (CSV + ROC figures + predictions), .../final_models/ (weights)
# Finished folds/models are cached: re-running this script only computes what is missing.
#
# Settings (environment variables, all optional):
#   RADGRAPH_PREPROCESS  v3 = the manuscript results (default here), v4 = body mask + soft-tissue window
#   RADGRAPH_DATA_DIR / RADGRAPH_OUTPUT_DIR / RADGRAPH_LOGS_DIR   data, outputs and logs (see DATA.md)
#   PY        Python interpreter (default: python on PATH)
#   N_JOBS    parallel jobs for preprocessing/extraction (default 8); TASK_JOBS per endpoint (default 6)
#
#   ./run_mateus_study.sh                                        # reproduce the manuscript (v3)
#   RADGRAPH_PREPROCESS=v4 RADGRAPH_OUTPUT_DIR=outputs_v4 ./run_mateus_study.sh
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
N_JOBS="${N_JOBS:-8}"
TASK_JOBS="${TASK_JOBS:-6}"
export RADGRAPH_PREPROCESS="${RADGRAPH_PREPROCESS:-v3}"
LOGDIR="${RADGRAPH_LOGS_DIR:-logs}"
mkdir -p "$LOGDIR"
log() { echo "=== $* $(date)"; }
log "preprocessing profile ${RADGRAPH_PREPROCESS} | python $($PY -c 'import sys; print(sys.executable)') | logs ${LOGDIR}"

log "preprocess/extract started"
$PY -u main.py --mode MATEUS_CV --task OS --stage preprocess --n_jobs "$N_JOBS" > "$LOGDIR/prepare_cohorts.log" 2>&1 &&
$PY -u main.py --mode MATEUS_CV --task OS --stage extract --n_jobs "$N_JOBS" >> "$LOGDIR/prepare_cohorts.log" 2>&1 \
  && log "preprocess/extract finished" || { log "preprocess/extract FAILED ($LOGDIR/prepare_cohorts.log)"; exit 1; }

for RULE in val_loss auc_gap; do
  for TASK in LR DM OS; do
    (
      for MODE in MATEUS_CV MATEUS_COHORT; do
        log "$MODE $TASK ($RULE) started"
        $PY -u main.py --mode $MODE --task $TASK --from_stage select --epoch_rule $RULE --n_jobs "$TASK_JOBS" \
            > "$LOGDIR/study_${MODE}_${TASK}_${RULE}.log" 2>&1 \
          && log "$MODE $TASK ($RULE) finished" || log "$MODE $TASK ($RULE) FAILED ($LOGDIR/study_${MODE}_${TASK}_${RULE}.log)"
      done
    ) &
  done
  wait
  $PY -u export_results.py > "$LOGDIR/export.log" 2>&1 && log "export ($RULE done) finished" || log "export FAILED ($LOGDIR/export.log)"
done
