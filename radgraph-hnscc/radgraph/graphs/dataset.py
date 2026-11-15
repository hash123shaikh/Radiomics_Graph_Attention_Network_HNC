"""
Graph datasets (+ clinical vectors) and the class-balancing sampler of Table S2.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.loader import DataLoader

from radgraph import config


# ─── Graph dataset ───────────────────────────────────────────────────────────

class RadGraphDataset(Dataset):
    """
    Wraps graphs; optionally attaches a clinical vector graph.clinical of shape (1, C).
    clinical : dict {pid: np.ndarray(C)} (already encoded + scaled) or None.
    A missing clinical row is an error (never silently zero-filled).
    """

    def __init__(self, graphs, clinical=None):
        self.graphs   = list(graphs)
        self.clinical = clinical
        if clinical is not None:
            missing = [g.patient_id for g in self.graphs if g.patient_id not in clinical]
            if missing:
                raise KeyError(f"No clinical vector for {len(missing)} patients: {missing[:5]}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        g = self.graphs[i]
        if self.clinical is not None:
            g = g.clone()
            g.clinical = torch.as_tensor(self.clinical[g.patient_id], dtype=torch.float).unsqueeze(0)
        return g

    @property
    def labels(self):
        return np.array([int(g.y.item()) for g in self.graphs])

    @property
    def patient_ids(self):
        return [g.patient_id for g in self.graphs]


# ─── Class balancing (Table S2 sampling strategies, authors' semantics) ─────

class ClassBalancedSampler(Sampler):
    """
    Re-drawn every epoch:
      'over'         minority oversampled (with replacement) to the majority size
      'under'        majority undersampled (without replacement) to the minority size
      'intermediate' both classes resampled to the mean of the two sizes ('mid' in the authors' code)
      'none'         plain shuffle
    """

    def __init__(self, labels, strategy, seed=0):
        self.labels   = np.asarray(labels)
        self.strategy = strategy
        self.rng      = np.random.default_rng(seed)
        pos, neg = np.where(self.labels == 1)[0], np.where(self.labels == 0)[0]
        self.minority, self.majority = (pos, neg) if len(pos) <= len(neg) else (neg, pos)
        if strategy == 'over':
            self.n_min, self.n_maj = len(self.majority), len(self.majority)
        elif strategy == 'under':
            self.n_min, self.n_maj = len(self.minority), len(self.minority)
        elif strategy == 'intermediate':
            m = int(round((len(self.minority) + len(self.majority)) / 2))
            self.n_min, self.n_maj = m, m
        elif strategy == 'none':
            self.n_min, self.n_maj = len(self.minority), len(self.majority)
        else:
            raise ValueError(strategy)

    @staticmethod
    def _draw(rng, idx, n):
        if n <= len(idx):
            return rng.choice(idx, size=n, replace=False)
        extra = rng.choice(idx, size=n - len(idx), replace=True)
        return np.concatenate([idx, extra])          # every sample at least once

    def __iter__(self):
        sel = np.concatenate([self._draw(self.rng, self.minority, self.n_min),
                              self._draw(self.rng, self.majority, self.n_maj)])
        self.rng.shuffle(sel)
        return iter(sel.tolist())

    def __len__(self):
        return self.n_min + self.n_maj


def make_loaders(train_ds, val_ds, test_ds, task, sampling=None, seed=None):
    g = config.get_gat_config(task)
    strategy = sampling or g['sampling_strategy']
    sampler = ClassBalancedSampler(train_ds.labels, strategy, seed=config.RANDOM_SEED if seed is None else seed)
    bs = g['batch_size']
    train = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=config.NUM_WORKERS)
    val   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=config.NUM_WORKERS)
    test  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=config.NUM_WORKERS)
    return train, val, test
