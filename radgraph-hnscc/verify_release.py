#!/usr/bin/env python3
"""
Verify the released models: re-apply every model ensemble to the MAASTRO (HN1) external cohort and
compare its AUC with the value recorded in its model_card.json (= the value reported in the paper).

This is the quickest way to reproduce the published external results. It needs only the MAASTRO
data (137 patients) and the release zip, and no training.

    python verify_release.py --download                      # fetch the release zip, then verify
    python verify_release.py --models_dir path/to/final_models
    python verify_release.py --download --only MATEUS_CV_LR  # a subset (substring match)

MAASTRO features are computed once with the preprocessing profile the models were trained with
(read from the model cards; v3 for v0.1.0-models) and cached under $RADGRAPH_OUTPUT_DIR/cache.
Exit code 0 = every ensemble reproduced its reported AUC within --tol.
"""

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = ('https://github.com/hash123shaikh/Radiomics_Graph_Attention_Network_HNC/releases/download/'
               'v0.1.0-models/radgraph-hnscc_final_models_v3.zip')


def download(dest_root):
    dest_root.mkdir(parents=True, exist_ok=True)
    z = dest_root / Path(RELEASE_URL).name
    if not z.exists():
        print(f"[download] {RELEASE_URL}")
        urllib.request.urlretrieve(RELEASE_URL, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(dest_root)
    return dest_root / 'radgraph-hnscc_final_models' / 'final_models'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models_dir', default=None, help='folder that holds <DESIGN>_<TASK>[_aucgap]/<inputs>/')
    ap.add_argument('--download', action='store_true', help='download and unzip the GitHub release first')
    ap.add_argument('--only', default=None, help='verify only model folders whose path contains this text')
    ap.add_argument('--n_jobs', type=int, default=4, help='parallel jobs for MAASTRO preprocessing')
    ap.add_argument('--tol', type=float, default=0.005, help='allowed |AUC difference|')
    a = ap.parse_args()

    from radgraph import config
    if a.download:
        models_dir = download(config.OUTPUT_DIR / 'release')
    else:
        models_dir = Path(a.models_dir) if a.models_dir else config.OUTPUT_DIR / 'final_models'
    cards = sorted(models_dir.rglob('model_card.json'))
    if a.only:
        cards = [c for c in cards if a.only in str(c.parent)]
    if not cards:
        raise SystemExit(f"No model_card.json under {models_dir}")

    profiles = {json.loads(c.read_text()).get('preprocess', 'v3') for c in cards}
    if len(profiles) != 1:
        raise SystemExit(f"Model cards mix preprocessing profiles {profiles}: verify them separately (--only)")
    config.set_preprocess(profiles.pop())
    print(f"[verify] {len(cards)} model ensembles | preprocessing {config.PREPROCESS} | output {config.OUTPUT_DIR}")

    from radgraph.data.loader import HNSCCDataLoader
    from radgraph.data.features import preprocess_and_save_all, extract_all_patients
    from radgraph.data.clinical import included_labels, encode_clinical
    from radgraph import inference as inf
    from radgraph.utils import safe_auc

    pids = HNSCCDataLoader('MAASTRO').eligible_patients_any_task()
    preprocess_and_save_all('MAASTRO', pids, n_jobs=a.n_jobs)
    extract_all_patients('MAASTRO', pids, n_jobs=a.n_jobs)
    from radgraph.data.features import feature_cache_valid
    pids = [p for p in pids if feature_cache_valid(p, 'MAASTRO')]

    rows, ok_all = [], True
    for c in cards:
        card = json.loads(c.read_text())
        config.set_mode(card['design'])
        config.set_task(card['task'])
        labels = included_labels('MAASTRO', card['task'], verbose=False)
        clinical, _ = encode_clinical('MAASTRO', harmonized=config.harmonized_clinical())
        use = [p for p in pids if p in labels]
        df = inf.predict_patients(inf.checkpoints_in(c.parent), use, clinical, labels)
        auc = safe_auc(df.label, df.prob)
        ref = float(card['external_test_auc'])
        ok = abs(auc - ref) <= a.tol
        ok_all &= ok
        name = str(c.parent.relative_to(models_dir))
        rows.append((name, len(use), ref, auc, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:45s} n={len(use):3d}  reported {ref:.4f}  reproduced {auc:.4f}")

    print(f"\n{sum(r[-1] for r in rows)}/{len(rows)} ensembles reproduced their reported external AUC "
          f"(tolerance {a.tol})")
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
