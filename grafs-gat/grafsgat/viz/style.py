"""Shared figure style: one validated categorical palette, a single-hue sequential ramp,
recessive grid/axes, text in ink colours (never the series colour)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8984'
GRID = '#e4e3df'
NEUTRAL = '#c9c8c2'

# categorical slots, fixed order (validated light-mode palette)
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
ACCENT = CAT[0]
SECOND = CAT[1]

# sequential (blue, light -> dark) and diverging (blue <-> gray <-> red)
SEQ = LinearSegmentedColormap.from_list(
    'seq_blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#256abf', '#86b6ef', '#f0efec', '#ec8f8f', '#c23a3a'])

FEATURE_CLASS_COLOUR = {   # six PyRadiomics classes -> first six slots, fixed
    'firstorder': CAT[0], 'glcm': CAT[1], 'glrlm': CAT[2],
    'glszm': CAT[3], 'gldm': CAT[4], 'ngtdm': CAT[5],
}


def apply():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': NEUTRAL, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.labelsize': 9,
        'xtick.color': INK_2, 'ytick.color': INK_2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'legend.fontsize': 8, 'font.size': 9,
        'lines.linewidth': 2, 'text.color': INK,
    })


def feature_class(name):
    parts = name.split('_')
    return parts[1] if len(parts) > 2 else 'other'


def short(name):
    """'original_glszm_ZoneEntropy' -> 'glszm ZoneEntropy'."""
    parts = name.split('_')
    return f'{parts[1]} {"_".join(parts[2:])}' if len(parts) > 2 else name


def save(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


apply()
