"""
Figures for the paper_workflow_v2 documents.
Reuses the manuscript figure code (Paper/manuscript/scripts/make_paper_assets.py) and adds
document-specific figures. Run with the GNNvenv interpreter:

    /home/radiomicsserver/anaconda3/envs/GNNvenv/bin/python make_figures.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
OUT = V2 / 'figures'
IDEA = V2.parent
MS = IDEA / 'Paper' / 'manuscript'
sys.path.insert(0, str(MS / 'scripts'))
import make_paper_assets as A          # noqa: E402  (sets style, paths, project imports)
import matplotlib.pyplot as plt        # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
PID = 'HN1260'


def copy_manuscript_figures():
    for f in ['fig2_topologies.png', 'fig_graphfs_workflow.png', 'fig_attention_ct.png',
              'fig_architecture_author.png', 'fig_roc_all.png', 'fig6_stability.png',
              'fig8_val_vs_test.png', 'fig5_auc_overview.png', 'fig_heatmap.png']:
        shutil.copy(MS / 'figures' / f, OUT / f)


def pipeline_overview():
    """Study pipeline (same drawing as the manuscript's original Figure 1)."""
    A.FIG = OUT
    A.fig_design()
    A.FIG = MS / 'figures'
    for ext in ('pdf',):
        p = OUT / f'fig1_study_design.{ext}'
        if p.exists():
            p.unlink()


ARCHIVE_V3 = IDEA / 'radgraph-hnscc' / 'outputs_archive_v3' / 'cache' / 'preprocessed'


def _load(pid, archive=False):
    from radgraph.data.features import preprocessed_path
    f = (ARCHIVE_V3 / f'{pid}.npz') if archive else preprocessed_path(pid)
    with np.load(f, allow_pickle=False) as d:
        return d['ct_array'], d['gtv_array'], d['supervoxel_labels']


def slic_steps():
    """Stage 2 before and after the v4 refinements on one patient."""
    from radgraph.data.supervoxels import body_mask
    ct, gtv, sv_old = _load(PID, archive=True)
    _, _, sv_new = _load(PID)
    z = int(np.argmax(gtv.reshape(gtv.shape[0], -1).sum(1)))
    img = np.clip(ct[z], -160, 240)
    rng = np.random.default_rng(1)
    cmap = ListedColormap(rng.uniform(0.15, 0.95, (512, 3)))
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.15))
    titles = ['CT and GTV', 'Version 3: whole box', 'Body mask', 'Version 4: body only']
    for ax, t in zip(axs, titles):
        ax.imshow(img, cmap='gray', interpolation='bilinear')
        ax.set_title(t, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax, lab in ((axs[1], sv_old), (axs[3], sv_new)):
        ax.imshow(np.ma.masked_where(lab[z] == 0, lab[z] % 512), cmap=cmap, alpha=0.55, interpolation='nearest')
    axs[2].contour(body_mask(ct)[z].astype(float), levels=[0.5], colors=[A.CAT[2]], linewidths=1.2)
    for ax in axs:
        ax.contour(gtv[z].astype(float), levels=[0.5], colors=[A.CAT[1]], linewidths=1.1)
    for ax, letter in zip(axs, 'abcd'):
        ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontsize=9, weight='bold', va='bottom')
    fig.tight_layout(pad=0.2, w_pad=0.3)
    A.FIG = OUT
    A.save(fig, 'slic_steps')
    A.FIG = MS / 'figures'
    (OUT / 'slic_steps.pdf').unlink(missing_ok=True)


def _stage2_stats(files, ct_files=None):
    """Per-patient supervoxel statistics over a set of preprocessed files."""
    n, vol, air, box, body = [], [], [], [], []
    from radgraph.data.supervoxels import body_mask
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            ct, sv = d['ct_array'], d['supervoxel_labels'].astype(np.int64)
        c = np.bincount(sv.ravel())
        ids = np.nonzero(c[1:])[0] + 1
        n.append(len(ids)); vol.extend((c[ids] / 1000.0).tolist())
        a = np.bincount(sv.ravel(), weights=(ct < -500).ravel().astype(float))
        air.append(float(np.mean(a[ids] / c[ids] > 0.5)))
        box.append(sv.size / 1e6)
        body.append(float(body_mask(ct).mean()))
    return dict(n=np.array(n), vol=np.array(vol), air=np.array(air), box=np.array(box), body=np.array(body))


def stats_and_histogram():
    """stats.json (numbers quoted in the documents) and the supervoxel-volume histogram (v3 vs v4)."""
    import glob, json
    import pandas as pd
    from radgraph import config
    S = {}
    v3 = _stage2_stats(sorted(glob.glob(str(ARCHIVE_V3 / '*.npz'))))
    v4 = _stage2_stats(sorted(glob.glob(str(config.PREPROCESSED_DIR / '*.npz'))))
    for tag, s in (('vThree', v3), ('vFour', v4)):
        S[f'{tag}Patients'] = str(len(s['n']))
        S[f'{tag}SvMin'] = str(int(s['n'].min())); S[f'{tag}SvMax'] = str(int(s['n'].max()))
        S[f'{tag}SvVolMed'] = f"{np.median(s['vol']):.1f}"
        S[f'{tag}SvTotal'] = f"{len(s['vol']):,}"
        S[f'{tag}AirMed'] = f"{100 * np.median(s['air']):.0f}"
        q = np.percentile(s['air'], [25, 75]); S[f'{tag}AirIqr'] = f"{100 * q[0]:.0f}–{100 * q[1]:.0f}"
        S[f'{tag}AirMax'] = f"{100 * s['air'].max():.0f}"
    S['boxMed'] = f"{np.median(v3['box']):.1f}"
    S['bodyFracMed'] = f"{100 * np.median(v4['body']):.0f}"
    # attention (reference configuration, fold 1, external patients)
    att = sorted(glob.glob(str(config.RUNS_DIR / 'MATEUS_CV_LR' / 'fold1' / 'attention' / 'table_s3__radiomic_star__img__s42_test.csv')))
    if att:
        d = pd.read_csv(att[0])
        rel = d.attention / d.groupby('patient_id').attention.transform('mean')
        lo, hi = np.percentile(rel, [1, 99])
        S['attDev'] = f"{100 * max(1 - lo, hi - 1):.1f}"
        g = d.groupby('patient_id').attention
        S['attRatio'] = f"{(g.max() / g.min()).median():.3f}"
        S['attPatients'] = str(d.patient_id.nunique())
    (OUT / 'stats.json').write_text(json.dumps(S, indent=1))
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    bins = np.linspace(0, np.percentile(v3['vol'], 99.5), 70)
    ax.hist(v3['vol'], bins=bins, color=A.NEUTRAL, label=f'version 3 (median {np.median(v3["vol"]):.1f} cm$^3$)')
    ax.hist(v4['vol'], bins=bins, color=A.CAT[0], alpha=0.75, label=f'version 4 (median {np.median(v4["vol"]):.1f} cm$^3$)')
    ax.set_xlabel('supervoxel volume (cm$^3$)'); ax.set_ylabel('supervoxels'); ax.legend(fontsize=6.5)
    fig.tight_layout()
    A.FIG = OUT
    A.save(fig, 'supervoxel_sizes')
    A.FIG = MS / 'figures'
    (OUT / 'supervoxel_sizes.pdf').unlink(missing_ok=True)
    return S


if __name__ == '__main__':
    copy_manuscript_figures()
    pipeline_overview()
    slic_steps()
    print(stats_and_histogram())
