"""
Graph-FS feature-similarity graph: nodes = radiomic features (coloured by PyRadiomics class),
edges = |r| above the adaptive threshold, selected representatives ringed and labelled,
communities (connected components / spectral clusters) drawn as faint groups.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from radgraph.viz.style import (FEATURE_CLASS_COLOUR, INK, INK_2, NEUTRAL, MUTED, feature_class,
                                short, save)


def plot_feature_graph(sel, path, title=''):
    info = sel['info']
    comps = info.get('components', [])
    edges = info.get('adjacency', [])
    nodes = [f for c in comps for f in c]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    comp_of = {f: i for i, c in enumerate(comps) for f in c}
    # layout: each community laid out separately, communities on a grid (readable for many components)
    pos, n = {}, len(comps)
    cols = int(np.ceil(np.sqrt(n)))
    for i, c in enumerate(comps):
        sub = G.subgraph(c)
        p = nx.spring_layout(sub, seed=1, k=0.9 / np.sqrt(max(len(c), 1))) if len(c) > 1 else {c[0]: np.zeros(2)}
        scale = 0.42 * np.sqrt(len(c) / max(len(x) for x in comps))
        cx, cy = i % cols, -(i // cols)
        for f, xy in p.items():
            pos[f] = np.array([cx, cy]) + scale * np.asarray(xy)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for i, c in enumerate(comps):           # faint community halos
        xy = np.array([pos[f] for f in c])
        ax.scatter(*xy.mean(0), s=(2600 * np.sqrt(len(c))), color='#f0efec', zorder=0, linewidths=0)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=NEUTRAL, width=0.6, alpha=0.8)
    selected = set(sel['selected'])
    colours = [FEATURE_CLASS_COLOUR.get(feature_class(f), MUTED) for f in nodes]
    sizes = [150 if f in selected else 40 for f in nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, node_color=colours, node_size=sizes,
                           edgecolors=[INK if f in selected else '#fcfcfb' for f in nodes],
                           linewidths=[2.0 if f in selected else 0.8 for f in nodes])
    for f in selected:
        if f in pos:
            ax.annotate(short(f), pos[f], xytext=(6, 6), textcoords='offset points', fontsize=7.5,
                        color=INK, fontweight='bold')
    handles = [Line2D([0], [0], marker='o', ls='', color=c, markersize=7, label=k)
               for k, c in FEATURE_CLASS_COLOUR.items()]
    handles.append(Line2D([0], [0], marker='o', ls='', markerfacecolor='white', markeredgecolor=INK,
                          markeredgewidth=2, markersize=9, label='selected'))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 1.0), title='feature class')
    sub = (f"|r| threshold {info.get('threshold', float('nan')):.2f} (mean + 0.3 SD), "
           f"{info.get('n_edges', len(edges))} edges, {len(comps)} groups, "
           f"{len(selected)} selected")
    ax.set_title(title + '\n', loc='left')
    ax.text(0, 1.0, sub, transform=ax.transAxes, fontsize=8.5, color=INK_2, va='bottom')
    return save(fig, path)
