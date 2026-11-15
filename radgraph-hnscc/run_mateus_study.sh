#!/usr/bin/env bash
# Full study, Mateus et al. (Sci Rep 2023) design: HNPET (4 Canadian centres) -> external MAASTRO.
#   1. preprocess + extract every eligible patient (cached)
#   2. LR, DM, OS in parallel: 5-fold CV and cohort split, checkpoint = lowest validation loss (primary)
#   3. same with Mateus' train-val AUC-gap checkpoint rule (sensitivity analysis)
#   4. export: outputs/results/ (CSV + ROC figures + predictions), outputs/final_models/ (weights)
# Finished folds/models are cached: re-running this script only computes what is missing.
cd "$(dirname "$0")"
PY="${PY:-python}"   # override: PY=/path/to/env/bin/python ./run_mateus_study.sh
log() { echo "=== $* $(date)"; }

log "preprocess/extract started"
$PY -u main.py --mode MATEUS_CV --task OS --stage preprocess --n_jobs 16 > logs/prepare_cohorts.log 2>&1 &&
$PY -u main.py --mode MATEUS_CV --task OS --stage extract --n_jobs 16 >> logs/prepare_cohorts.log 2>&1 \
  && log "preprocess/extract finished" || { log "preprocess/extract FAILED (logs/prepare_cohorts.log)"; exit 1; }

for RULE in val_loss auc_gap; do
  for TASK in LR DM OS; do
    (
      for MODE in MATEUS_CV MATEUS_COHORT; do
        log "$MODE $TASK ($RULE) started"
        $PY -u main.py --mode $MODE --task $TASK --from_stage select --epoch_rule $RULE --n_jobs 6 \
            > logs/study_${MODE}_${TASK}_${RULE}.log 2>&1 \
          && log "$MODE $TASK ($RULE) finished" || log "$MODE $TASK ($RULE) FAILED (logs/study_${MODE}_${TASK}_${RULE}.log)"
      done
    ) &
  done
  wait
  $PY -u export_results.py > logs/export.log 2>&1 && log "export ($RULE done) finished" || log "export FAILED (logs/export.log)"
done
