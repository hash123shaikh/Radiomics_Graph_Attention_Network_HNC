"""
Provenance record written next to every run, so that each result can be traced to the exact code,
settings and software that produced it.

    write_provenance(folder, argv)  ->  <folder>/provenance.json
"""

import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from grafsgat import config

PACKAGES = ['torch', 'torch-geometric', 'numpy', 'scipy', 'scikit-learn', 'pyradiomics', 'SimpleITK',
            'scikit-image', 'pandas', 'mrmr-selection', 'networkx', 'pydicom']


def _git(*args):
    try:
        return subprocess.run(['git', *args], cwd=config.BASE_DIR, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def _versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def collect(argv=None):
    import torch
    return {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'command': ' '.join(argv if argv is not None else sys.argv),
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_dirty': bool(_git('status', '--porcelain', '--untracked-files=no')),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'device': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'),
        'cuda': torch.version.cuda,
        'packages': _versions(),
        'settings': {
            'mode': config.MODE, 'task': config.TASK, 'epoch_rule': config.EPOCH_RULE,
            'preprocess': config.PREPROCESS, 'preprocess_version': config.PREPROCESS_VERSION,
            'use_body_mask': config.USE_BODY_MASK, 'slic_hu_window': config.SLIC_HU_WINDOW,
            'pyradiomics': config.PYRADIOMICS_SETTINGS, 'fs_k': config.FS_K,
            'seeds': config.BENCHMARK_SEEDS, 'random_seed': config.RANDOM_SEED, 'n_folds': config.N_FOLDS,
            'gat': config.get_gat_config(config.TASK),
        },
        'paths': {'data': str(config.DATA_DIR), 'output': str(config.OUTPUT_DIR)},
    }


def write_provenance(folder, argv=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rec = collect(argv)
    (folder / 'provenance.json').write_text(json.dumps(rec, indent=1, default=str))
    return rec
