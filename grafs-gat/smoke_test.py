#!/usr/bin/env python3
"""
Smoke test: runs every stage of the pipeline on SYNTHETIC data, so an installation can be checked
without downloading the patient images. It takes about 1-2 minutes on a CPU and writes only to a
temporary folder.

    python smoke_test.py

Stages exercised: preprocessing + SLIC supervoxels (v3 and v4) -> PyRadiomics features -> all
9 feature selectors -> stability metrics -> the 4 graph topologies -> GAT training and inference
-> DeLong / Holm / bootstrap CI.
"""

import os
import sys
import tempfile
import time

TMP = tempfile.mkdtemp(prefix='grafsgat_smoke_')
os.environ['GRAFS_OUTPUT_DIR'] = os.path.join(TMP, 'outputs')
os.environ['GRAFS_LOGS_DIR'] = os.path.join(TMP, 'logs')

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

RESULTS = []


def step(name):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            try:
                out = fn(*a, **k)
                RESULTS.append((name, True, f'{time.time() - t:.1f}s'))
                print(f"  PASS  {name}  ({time.time() - t:.1f}s)")
                return out
            except Exception as e:
                RESULTS.append((name, False, f'{type(e).__name__}: {e}'))
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
                raise
        return run
    return deco


def synthetic_patient(seed, size=(48, 96, 96)):
    """CT-like volume (HU) with a body ellipse, soft tissue texture and a spherical 'tumour'."""
    import SimpleITK as sitk
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(size)
    c = np.array(size) / 2
    body = ((y - c[1]) / 44) ** 2 + ((x - c[2]) / 40) ** 2 < 1
    ct = np.full(size, -1000.0)
    ct[body] = 40 + 25 * rng.standard_normal(body.sum())
    r = 7 + 3 * rng.random()
    gtv = ((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) < r ** 2
    ct[gtv] = 70 + (20 + 20 * rng.random()) * rng.standard_normal(gtv.sum())
    img = sitk.GetImageFromArray(ct.astype(np.int16))
    img.SetSpacing((1.0, 1.0, 2.0))
    msk = sitk.GetImageFromArray(gtv.astype(np.uint8))
    msk.CopyInformation(img)
    return img, msk


@step('preprocessing + supervoxels + radiomics (v3 and v4)')
def stage_features():
    from grafsgat import config
    from grafsgat.data.features import preprocess_images, SupervoxelFeatureExtractor, load_feature_dict
    fds = {}
    for profile in ('v3', 'v4'):
        config.set_preprocess(profile)
        for i in range(3 if profile == 'v3' else 1):
            pid = f'SYN{i}_{profile}'
            ct, m = synthetic_patient(i)
            preprocess_images(pid, ct, m, 'synthetic', 'NEW')
            n = SupervoxelFeatureExtractor().extract_patient(pid, 'NEW')
            assert n >= 20, f'only {n} supervoxels'
            fds[pid] = load_feature_dict(pid)
    assert len(fds['SYN0_v3']['feature_names']) == 93, len(fds['SYN0_v3']['feature_names'])
    return fds


@step('9 feature selectors + stability metrics')
def stage_selection(names):
    from grafsgat.selection.methods import run_selector
    from grafsgat.selection.stability import kuncheva, jaccard
    from grafsgat import config
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(rng.standard_normal((n, len(names))), columns=names)
    X.iloc[:, 1:20] += X.iloc[:, [0]].values * 2              # a correlated block, as in radiomics
    y = (X.iloc[:, 0] + rng.standard_normal(n) > 0.5).astype(int).values
    sel = {}
    for m in config.FS_METHODS:
        r = run_selector(m, X, y, task='LR', k=6, seed=0, n_jobs=1)
        assert len(r.selected) > 0, m
        sel[m] = r.selected
    a, b = sel['graph_fs_spectral'], sel['mrmr']
    kuncheva(set(a), set(b), len(names)), jaccard(set(a), set(b))
    return sel


@step('graphs (4 topologies) + GAT training and inference')
def stage_gat(fds):
    import torch
    from torch_geometric.loader import DataLoader
    from grafsgat import config
    from grafsgat.graphs.builder import GTVScaler, build_patient_graph
    from grafsgat.models.gat import GRAFSGAT
    torch.manual_seed(0)
    keys = [k for k in fds if k.endswith('v3')]
    names = fds[keys[0]]['feature_names']
    idx = list(range(6))
    sc = GTVScaler().fit(np.vstack([fds[k]['gtv'][idx] for k in keys]))
    for topo in config.TOPOLOGIES:
        graphs = [build_patient_graph(fds[k], idx, sc, topo, i % 2, k) for i, k in enumerate(keys * 4)]
        assert graphs[0].num_nodes == config.N_SUPERVOXELS_SELECTED + 1, graphs[0].num_nodes
        for task in ('LR', 'DM'):
            model = GRAFSGAT(n_node_features=len(idx), task=task)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(3):
                for b in DataLoader(graphs, batch_size=4, shuffle=True):
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(b), b.y.float().view(-1))
                    opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(next(iter(DataLoader(graphs, batch_size=len(graphs))))))
            assert torch.isfinite(p).all()
    return names


@step('statistics (DeLong, Holm, bootstrap CI)')
def stage_stats():
    from grafsgat.stats import delong_test, holm
    from grafsgat.utils import bootstrap_auc_ci
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p1, p2 = y + rng.standard_normal(200), rng.standard_normal(200)
    a1, a2, z, p = delong_test(y, p1, p2)
    assert a1 > 0.6 and p < 0.05, (a1, p)
    holm([0.01, 0.04, 0.2])
    lo, hi = bootstrap_auc_ci(y, p1, n_bootstrap=200)
    assert lo < a1 < hi


def main():
    print(f"GRAFS-GAT smoke test (synthetic data, temporary folder {TMP})")
    try:
        fds = stage_features()
        stage_selection(fds['SYN0_v3']['feature_names'])
        stage_gat(fds)
        stage_stats()
    except Exception:
        pass
    import shutil
    shutil.rmtree(TMP, ignore_errors=True)
    ok = all(r[1] for r in RESULTS) and len(RESULTS) == 4
    print(f"\n{'ALL PASSED' if ok else 'FAILED'}: {sum(r[1] for r in RESULTS)}/4 stages")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
