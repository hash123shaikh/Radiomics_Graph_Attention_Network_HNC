"""
Predict LR / DM / OS risk with the SAVED models (no training).

Which models
  default            the configuration chosen on validation AUC in that run (imaging only;
                     --inputs imaging+clinical for the clinical variant): the full ensemble of
                     5 folds x 3 seeds (MATEUS_CV) or 3 seeds (MATEUS_COHORT)
  --config M__T      any other configuration, e.g. graph_fs_spectral__radiomic_star
  --models DIR       every .pth in a folder, e.g. outputs/final_models/LR_MATEUS_CV/imaging

Which patients
  --dataset MAASTRO [--patients HN1004 HN1006 ...]     patients of a known cohort (cached or processed now)
  --ct CT.nii.gz --mask SEG.nii.gz --id NEW001 [--label 1]   a new patient (imaging-only models)

Examples
  python predict.py --task LR --dataset MAASTRO
  python predict.py --task OS --design MATEUS_COHORT --inputs imaging+clinical --dataset MAASTRO
  python predict.py --task DM --config mrmr__spatial_star --dataset MAASTRO --patients HN1004 HN1006
  python predict.py --task LR --ct new/CT.nii.gz --mask new/GTV.nii.gz --id NEW001

Output: outputs/predictions/<TASK>_<DESIGN>_<method>__<topology>__<inputs>_<who>.csv
  prob = mean predicted probability of the event within the task horizon (LR/DM 2 y, OS 4 y),
  prob_sd = spread across the ensemble members, predicted_high_risk = prob >= mean validation
  Youden threshold, used_in_training = the patient was in the training/validation data of the models.
"""

import argparse
from pathlib import Path

from radgraph import config
from radgraph import inference as inf


def main():
    ap = argparse.ArgumentParser(description='RadGraph-FS inference with saved models', epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True, choices=config.TASKS)
    ap.add_argument('--design', default='MATEUS_CV', choices=list(config.MODES))
    ap.add_argument('--epoch_rule', default='val_loss', choices=['val_loss', 'auc_gap'])
    ap.add_argument('--inputs', default='imaging', choices=['imaging', 'imaging+clinical'])
    ap.add_argument('--config', default=None, help='method__topology (default: the configuration chosen on validation)')
    ap.add_argument('--models', default=None, help='folder of .pth checkpoints (overrides --design/--config)')
    ap.add_argument('--dataset', default=None, choices=list(config.DATASETS))
    ap.add_argument('--patients', nargs='+', default=None)
    ap.add_argument('--ct', default=None)
    ap.add_argument('--mask', default=None)
    ap.add_argument('--label', type=int, default=1, help='label of the primary GTV in --mask')
    ap.add_argument('--id', default=None, help='identifier for a new patient (--ct/--mask)')
    ap.add_argument('--per_model', action='store_true', help='also write every ensemble member\'s probability')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    config.set_mode(a.design)
    config.set_task(a.task)
    clin_flag = a.inputs == 'imaging+clinical'
    if a.models:
        ckpts, tag = inf.checkpoints_in(a.models), Path(a.models).name
    else:
        if a.config:
            method, topology = a.config.split('__')[:2]
        else:
            method, topology, clin_flag = inf.chosen_config(a.task, a.design, a.inputs, a.epoch_rule)
        ckpts = inf.find_checkpoints(a.task, a.design, method, topology, clin_flag, a.epoch_rule)
        tag = f"{method}__{topology}__{'img+clin' if clin_flag else 'img'}"
    print(f"[predict] {a.task} | {len(ckpts)} checkpoints | {tag}")

    labels, clinical, who = None, None, None
    if a.ct or a.mask:
        if not (a.ct and a.mask and a.id):
            raise SystemExit("--ct, --mask and --id are needed together")
        pid = f'NEW_{a.id}'
        inf.ensure_features_from_files(pid, a.ct, a.mask, a.label)
        pids, who = [pid], a.id
    elif a.dataset:
        from radgraph.data.loader import HNSCCDataLoader
        from radgraph.data.clinical import included_labels, encode_clinical
        pids = a.patients or HNSCCDataLoader(a.dataset).eligible_patients_any_task()
        ok = []
        for p in pids:
            try:
                inf.ensure_features(p, a.dataset)
                ok.append(p)
            except Exception as e:
                print(f"  skipped {p}: {type(e).__name__}: {e}")
        pids, who = ok, a.dataset if not a.patients else f"{a.dataset}_{len(ok)}pts"
        if config.task_available(a.dataset, a.task):
            labels = included_labels(a.dataset, a.task, verbose=False)
        clinical, _ = encode_clinical(a.dataset, harmonized=config.harmonized_clinical())
    else:
        raise SystemExit("Give --dataset (optionally --patients) or --ct/--mask/--id")

    df = inf.predict_patients(ckpts, pids, clinical, labels, inf.training_ids(ckpts), a.per_model)
    out = Path(a.out) if a.out else config.OUTPUT_DIR / 'predictions' / f"{a.task}_{a.design}_{tag}_{who}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df[['patient_id', 'prob', 'prob_sd', 'predicted_high_risk'] + (['label'] if labels else [])]
          .head(15).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    if labels:
        from radgraph.utils import safe_auc
        e = df[df.label.notna() & ~df.used_in_training]
        if e.label.nunique() == 2:
            print(f"AUC on {len(e)} labelled patients not used in training: {safe_auc(e.label, e.prob):.3f}")
    print(f"-> {out}")


if __name__ == '__main__':
    main()
