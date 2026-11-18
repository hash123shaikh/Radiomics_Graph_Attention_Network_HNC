"""
Patient-level graph figures (what the GAT actually sees):

  left  : axial CT slice through the GTV centroid; GTV contour; the graph's supervoxels
          outlined and filled by GAT attention (GTV read-out -> supervoxel); edges to the GTV
  right : the graph itself — GTV hub, node angle = in-plane direction from the GTV,
          radius = 3-D distance (mm); edge width / node colour = attention; kNN edges grey
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from grafsgat import config
from grafsgat.data.features import preprocessed_path
from grafsgat.viz.style import SEQ, INK, INK_2, NEUTRAL, SECOND, save


def _pick_examples(att, n=4):
    p = att.groupby('patient_id').agg(label=('label', 'first'), prob=('prob', 'first'))
    picks = []
    pos, neg = p[p.label == 1].sort_values('prob'), p[p.label == 0].sort_values('prob')
    if len(pos):
        picks += [(pos.index[-1], 'highest-risk event'), (pos.index[0], 'lowest-risk event')]
    if len(neg):
        picks += [(neg.index[-1], 'highest-risk non-event'), (neg.index[0], 'lowest-risk non-event')]
    seen, out = set(), []
    for pid, why in picks:
        if pid not in seen:
            seen.add(pid); out.append((pid, why))
    return out[:n]


def plot_patient_graph(task, method, topology, pid, att, path, note=''):
    g = torch.load(config.graph_dir(task, method, topology) / f'{pid}.pt', weights_only=False)
    a = att[att.patient_id == pid].set_index('node')['attention']
    with np.load(preprocessed_path(pid), allow_pickle=False) as d:
        ct, gtv, sv = d['ct_array'], d['gtv_array'], d['supervoxel_labels']
    pos = g.pos.numpy()                 # (z, y, x) in mm == voxels (1 mm grid, cropped volume)
    z = int(np.clip(round(pos[0, 0]), 0, ct.shape[0] - 1))
    # colour = attention within this patient's range (GAT attention is often close to uniform 1/K)
    lo, hi = (float(a.min()), float(a.max())) if len(a) else (0.0, 1.0)
    norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-6)
    label, prob = int(g.y.item()), float(att[att.patient_id == pid].prob.iloc[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={'width_ratios': [1.1, 1]})
    # ── left: anatomy ──
    ax1.grid(False)
    ax1.imshow(np.clip(ct[z], -250, 250), cmap='gray', interpolation='nearest')
    overlay = np.full(ct[z].shape, np.nan)
    for node, sv_id in enumerate(g.sv_ids.numpy()):
        if node == 0:
            continue
        m = sv[z] == sv_id
        if m.any():
            overlay[m] = a.get(node, 0.0)
            ax1.contour(m, levels=[0.5], colors=['white'], linewidths=0.4)
    ax1.imshow(overlay, cmap=SEQ, norm=norm, alpha=0.65, interpolation='nearest')
    if gtv[z].any():
        ax1.contour(gtv[z], levels=[0.5], colors=[SECOND], linewidths=2)
    for node in range(1, len(pos)):
        ax1.plot([pos[0, 2], pos[node, 2]], [pos[0, 1], pos[node, 1]], color='white', lw=0.5, alpha=0.5)
        in_plane = abs(pos[node, 0] - z) <= 5
        ax1.scatter(pos[node, 2], pos[node, 1], s=28 if in_plane else 10, color=SEQ(norm(a.get(node, 0))),
                    edgecolors='white', linewidths=0.6, alpha=1 if in_plane else 0.6, zorder=3)
    ax1.scatter(pos[0, 2], pos[0, 1], s=90, marker='*', color=SECOND, edgecolors='white', zorder=4)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f'Axial CT through GTV (slice {z}); orange = GTV, fill = attention', loc='left', fontsize=9.5)

    # ── right: graph ──
    ax2.grid(False); ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    d3 = np.linalg.norm(pos - pos[0], axis=1)
    ang = np.arctan2(-(pos[:, 1] - pos[0, 1]), pos[:, 2] - pos[0, 2])
    xy = np.c_[d3 * np.cos(ang), d3 * np.sin(ang)]
    xy[0] = 0
    ei = g.edge_index.numpy()
    for s_, t_ in ei.T:
        if s_ < t_ and s_ != 0 and t_ != 0:                       # kNN edges between supervoxels
            ax2.plot(*xy[[s_, t_]].T, color=NEUTRAL, lw=0.6, zorder=1)
    for node in range(1, len(pos)):
        w = a.get(node, 0.0)
        ax2.plot(*xy[[0, node]].T, color=SEQ(norm(w)), lw=0.6 + 5 * norm(w), alpha=0.9, zorder=2)
        ax2.scatter(*xy[node], s=60, color=SEQ(norm(w)), edgecolors=INK, linewidths=0.5, zorder=3)
        ax2.text(xy[node, 0], xy[node, 1], f' {node}', fontsize=6.5, color=INK_2, va='center')
    ax2.scatter(0, 0, s=320, marker='*', color=SECOND, edgecolors=INK, zorder=4)
    ax2.text(0, 0, '  GTV', fontsize=8.5, fontweight='bold', va='center')
    r = max(d3.max(), 1)
    for ring in np.linspace(r / 3, r, 3):
        ax2.add_patch(plt.Circle((0, 0), ring, fill=False, color=NEUTRAL, lw=0.6, ls=':'))
        ax2.text(0, ring, f'{ring:.0f} mm', fontsize=6.5, color=INK_2, ha='center', va='bottom')
    ax2.set_aspect('equal'); ax2.set_xlim(-1.15 * r, 1.15 * r); ax2.set_ylim(-1.15 * r, 1.15 * r)
    ax2.set_title(f'Graph ({topology}): radius = distance from GTV, width/colour = attention',
                  loc='left', fontsize=9.5)
    fig.colorbar(ScalarMappable(norm=norm, cmap=SEQ), ax=ax2, fraction=0.04, label=f'attention to GTV read-out (range {lo:.3f}-{hi:.3f}; uniform 1/(K+1) = {1 / (len(a) + 1):.3f})')
    fig.suptitle(f"{pid}  |  {'LR event' if label else 'no LR'} within 2 y  |  predicted risk {prob:.2f}"
                 f"{'  |  ' + note if note else ''}  |  {method}", x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    return save(fig, path, dpi=170)


def plot_patient_examples(task, method, topology, att, out_dir, prefix='', n=4):
    paths = []
    for pid, why in _pick_examples(att, n):
        try:
            paths.append(plot_patient_graph(task, method, topology, pid, att,
                                            out_dir / f'{prefix}_{pid}.png', note=why))
        except Exception as e:
            print(f"  [patient graph] {pid}: {type(e).__name__}: {e}")
    # attention vs distance summary for all test patients
    if not att.empty:
        att = att.copy()
        att['dist_mm'] = np.nan
        for pid, grp in att.groupby('patient_id'):
            g = torch.load(config.graph_dir(task, method, topology) / f'{pid}.pt', weights_only=False)
            pos = g.pos.numpy()
            att.loc[grp.index, 'dist_mm'] = np.linalg.norm(pos[grp.node.values] - pos[0], axis=1)
        att['att_rel'] = att.attention / att.groupby('patient_id').attention.transform('mean')
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for lab, col, name in ((0, NEUTRAL, 'no LR'), (1, SECOND, 'LR event')):
            s = att[att.label == lab]
            ax.scatter(s.dist_mm, s.att_rel, s=8, color=col, alpha=0.5, label=name)
        ax.axhline(1, color=INK_2, ls='--', lw=0.8)
        ax.set_xlabel('Supervoxel distance from GTV centroid (mm)')
        ax.set_ylabel('Attention / patient mean')
        ax.legend(); ax.set_title('Where does the GTV read-out attend? (test patients)', loc='left')
        paths.append(save(fig, out_dir / f'{prefix}_attention_vs_distance.png'))
    return paths
