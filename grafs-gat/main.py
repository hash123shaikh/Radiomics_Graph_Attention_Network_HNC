"""
GRAFS-GAT pipeline: LR / DM / OS prediction in HNSCC with a graph attention network, comparing
feature-selection methods x graph topologies for discrimination and reproducibility, with the
validation design of Mateus et al. (Sci Rep 2023): train/validate on the 4 Canadian centres
(HNPET), test on the external MAASTRO cohort.

Stages (in order)
  preprocess  CT + primary GTV -> 1 mm, 5 cm box, SLIC supervoxels        (per cohort, cached)
  extract     PyRadiomics features for GTV + supervoxels                 (per cohort, cached)
  select      every FS method on the TRAINING split + selection figures        (per fold)
  stability   bootstrap stability + cross-centre reproducibility                (per fold)
  benchmark   graphs + GAT (methods x topologies x inputs x seeds) + RF baselines (per fold)
  summary     seed-ensembled metrics, DeLong/Holm, figures, report.html      (per fold + cross-fold)

Examples
  python main.py --mode MATEUS_CV --task LR --all --n_jobs 12            # everything, 5 folds + summary
  python main.py --mode MATEUS_CV --task DM --from_stage select          # all folds, cached stages skipped
  python main.py --mode MATEUS_CV --task OS --from_stage select --fold 2 # one fold only
  python main.py --mode MATEUS_COHORT --task LR --from_stage select --epoch_rule auc_gap
  python export_results.py                                               # all tasks -> outputs/results/
  python predict.py --task LR --design MATEUS_CV --dataset MAASTRO       # inference with saved models

Modes: MATEUS_CV (5-fold CV on HNPET -> MAASTRO) | MATEUS_COHORT (HGJ+CHUS / HMR+CHUM -> MAASTRO)
Tasks: LR, DM (2-year horizon), OS (4-year horizon)
Each run is written to outputs/runs/<MODE>_<TASK>/ (open report.html); MATEUS_CV writes one
sub-folder per fold (fold1..fold5) plus the cross-fold summary in the mode folder.
--epoch_rule auc_gap (Mateus et al. checkpoint rule) writes to outputs/runs/<MODE>_<TASK>_aucgap/.
"""

import argparse
import time

from grafsgat import config

STAGES = ['preprocess', 'extract', 'select', 'stability', 'benchmark', 'summary']
PER_FOLD = ['select', 'stability', 'benchmark', 'summary']     # repeated for every fold in 'cv' modes


def cohort_patients(dataset, limit=None):
    from grafsgat.data.loader import HNSCCDataLoader
    ids = HNSCCDataLoader(dataset).eligible_patients_any_task()
    return ids[:limit] if limit else ids


def run_stage(stage, a):
    k = 'default' if a.k is None else (None if a.k == 'native' else int(a.k))
    if stage == 'preprocess':
        from grafsgat.data.features import preprocess_and_save_all
        for ds in config.datasets_in_mode():
            preprocess_and_save_all(ds, cohort_patients(ds, a.limit), force=a.force, n_jobs=a.n_jobs)
    elif stage == 'extract':
        from grafsgat.data.features import extract_all_patients
        for ds in config.datasets_in_mode():
            extract_all_patients(ds, cohort_patients(ds, a.limit), n_jobs=a.n_jobs, force=a.force)
    elif stage == 'select':
        from grafsgat.selection.runner import run_selection
        run_selection(a.task, a.methods, k, force=a.force)
    elif stage == 'stability':
        from grafsgat.selection.runner import run_stability
        run_stability(a.task, a.methods, k, B=a.bootstrap, n_jobs=max(1, a.n_jobs))
    elif stage == 'benchmark':
        from grafsgat.experiments.benchmark import run_grid
        variants = {'imaging': (False,), 'clinical': (True,), 'both': (False, True)}[a.variant]
        run_grid(a.task, a.methods, a.topologies, variants, a.seeds, a.epochs, force=a.force)
    elif stage == 'summary':
        from grafsgat.experiments.benchmark import summarize
        summarize(a.task)


def main():
    ap = argparse.ArgumentParser(description='GRAFS-GAT pipeline', epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', '--dataset', dest='mode', default=config.MODE,
                    choices=list(config.MODES) + list(config.MODE_ALIASES))
    ap.add_argument('--task', default='LR', choices=config.TASKS)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--stage', choices=STAGES)
    g.add_argument('--from_stage', choices=STAGES)
    g.add_argument('--all', action='store_true')
    ap.add_argument('--methods', nargs='+', default=None, choices=config.FS_METHODS)
    ap.add_argument('--topologies', nargs='+', default=None, choices=config.TOPOLOGIES)
    ap.add_argument('--variant', default='both', choices=['imaging', 'clinical', 'both'])
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--k', default=None, help="features per method (int) or 'native' (default: config.FS_K)")
    ap.add_argument('--bootstrap', type=int, default=None, help='stability bootstrap resamples')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--n_jobs', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None, help='first N patients only (testing)')
    ap.add_argument('--force', action='store_true', help='recompute cached results')
    ap.add_argument('--fold', type=int, default=None,
                    help=f'cross-validation modes: run this fold only (1..{config.N_FOLDS}; default all folds)')
    ap.add_argument('--epoch_rule', default=config.EPOCH_RULE, choices=['val_loss', 'auc_gap'],
                    help="checkpoint rule: lowest val loss (paper) or Mateus et al.'s train-val AUC gap rule")
    ap.add_argument('--preprocess', default=config.PREPROCESS, choices=list(config.PREPROCESS_PROFILES),
                    help="preprocessing profile: v4 = body mask + soft-tissue window (manuscript), v3 = earlier "
                         "analysis / release v0.1.0-models (default: $GRAFS_PREPROCESS or %(default)s)")
    a = ap.parse_args()

    config.set_preprocess(a.preprocess)
    config.set_mode(a.mode)
    config.EPOCH_RULE = a.epoch_rule
    config.set_task(a.task)
    cv = config.split_scheme() == 'cv'
    if a.fold is not None and not cv:
        raise SystemExit(f"--fold only applies to cross-validation modes, not {config.MODE}")
    if not config.mode_task_available(a.task):
        raise SystemExit(f"Task {a.task} is not available for every cohort of mode {config.MODE}")
    stages = STAGES if a.all else STAGES[STAGES.index(a.from_stage):] if a.from_stage else [a.stage]
    config.print_config(a.task)
    from grafsgat.provenance import write_provenance
    write_provenance(config.mode_dir(a.task))
    t0 = time.time()
    folds = ([a.fold] if a.fold else list(range(1, config.N_FOLDS + 1))) if cv else [None]

    def stage(st, fold=None):
        config.set_fold(fold)
        ts = time.time()
        print(f"\n{'=' * 70}\n  Stage: {st}{f'  (fold {fold}/{config.N_FOLDS})' if fold else ''}\n{'=' * 70}")
        run_stage(st, a)
        print(f"  [{st}{f' fold {fold}' if fold else ''} done in {time.time() - ts:.0f}s]")

    for st in [s for s in stages if s not in PER_FOLD]:          # preprocess / extract: once
        stage(st)
    fold_stages = [s for s in stages if s in PER_FOLD]
    for f in folds:                                             # one complete run per fold
        for st in fold_stages:
            stage(st, f)
    config.set_fold(None)
    if cv and 'summary' in stages:
        done = all((config.mode_dir(a.task) / f'fold{k}' / 'results_summary.csv').exists()
                   for k in range(1, config.N_FOLDS + 1))
        if done:
            from grafsgat.experiments.cv import summarize_cv
            print(f"\n{'=' * 70}\n  Cross-fold summary\n{'=' * 70}")
            summarize_cv(a.task)
        else:
            print("\n[cv] cross-fold summary skipped: not every fold has finished")
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min -> {config.mode_dir(a.task)}")


if __name__ == '__main__':
    main()
