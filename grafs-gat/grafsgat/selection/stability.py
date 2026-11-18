"""
Stability / reproducibility of feature selection (Moradmand et al. metrics, adapted).

  within-cohort stability : B stratified bootstrap resamples of the TRAINING set
  cross-centre reproducibility : selection run separately on each centre's training patients

Metrics (mean over all pairs of runs):
  jaccard   |A n B| / |A u B|
  dice      2|A n B| / (|A| + |B|)                      (Dice similarity index, DSI)
  kuncheva  (r p - k^2) / (k (p - k)),  r = |A n B|      (chance-corrected; equal sizes only)
  kendall_w concordance of the full feature rankings across runs (0..1)
  core80    fraction of the k selected features that are selected in >= 80% of runs
"""

from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from grafsgat import config
from grafsgat.selection.methods import run_selector, FIXED


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def dice(a, b):
    a, b = set(a), set(b)
    return 2 * len(a & b) / (len(a) + len(b)) if (a or b) else 1.0


def kuncheva(a, b, p):
    a, b = set(a), set(b)
    k = len(a)
    if k != len(b) or k == 0 or k >= p:
        return np.nan
    r = len(a & b)
    return (r * p - k * k) / (k * (p - k))


def kendall_w(rankings, features):
    """rankings: list of ordered feature lists (each a permutation of a superset of `features`)."""
    m, n = len(rankings), len(features)
    if m < 2 or n < 2:
        return np.nan
    pos = np.zeros((m, n))
    for i, r in enumerate(rankings):
        idx = {f: j for j, f in enumerate([f for f in r if f in set(features)])}
        pos[i] = [idx.get(f, n - 1) + 1 for f in features]
    R = pos.sum(0)
    S = ((R - R.mean()) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def pairwise(sets, p):
    pairs = list(combinations(range(len(sets)), 2))
    if not pairs:
        return {'jaccard': np.nan, 'dice': np.nan, 'kuncheva': np.nan}
    return {
        'jaccard' : float(np.mean([jaccard(sets[i], sets[j]) for i, j in pairs])),
        'dice'    : float(np.mean([dice(sets[i], sets[j]) for i, j in pairs])),
        'kuncheva': float(np.nanmean([kuncheva(sets[i], sets[j], p) for i, j in pairs]))
                    if any(len(sets[i]) == len(sets[j]) for i, j in pairs) else np.nan,
    }


def frequency(sets):
    f = pd.Series([x for s in sets for x in s]).value_counts() / len(sets)
    return f.sort_values(ascending=False)


def _summary(method, sets, rankings, p, label):
    freq = frequency(sets)
    k = int(np.median([len(s) for s in sets]))
    common = set.intersection(*[set(r) for r in rankings]) if rankings else set()
    return {'method': method, 'setting': label, 'n_runs': len(sets), 'k_median': k,
            **pairwise(sets, p),
            'kendall_w': kendall_w(rankings, sorted(common)),
            'core80': float((freq >= 0.8).sum() / max(k, 1))}


def _one_bootstrap(method, X, y, task, k, seed):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], (y == c).sum(), replace=True) for c in (0, 1)])
    Xb = X.iloc[idx].reset_index(drop=True)
    r = run_selector(method, Xb, y[idx], task, k, seed=seed, n_jobs=1)
    return r.selected, r.ranking


def bootstrap_stability(X, y, methods, task, k='default', B=None, n_jobs=8):
    """Returns (summary DataFrame, {method: frequency Series}, {method: [selected sets]})."""
    B = B or config.STABILITY_N_BOOTSTRAP
    y = np.asarray(y).astype(int)
    rows, freqs, runs = [], {}, {}
    for m in methods:
        if m in FIXED:
            continue
        out = Parallel(n_jobs=n_jobs)(delayed(_one_bootstrap)(m, X, y, task, k, 1000 + b) for b in range(B))
        sets, ranks = [o[0] for o in out], [o[1] for o in out]
        rows.append(_summary(m, sets, ranks, X.shape[1], f'bootstrap B={B}'))
        freqs[m], runs[m] = frequency(sets), sets
        print(f"  [stability] {m:18s} jaccard={rows[-1]['jaccard']:.3f} "
              f"kuncheva={rows[-1]['kuncheva']:.3f} kendallW={rows[-1]['kendall_w']:.3f}")
    return pd.DataFrame(rows), freqs, runs


def centre_reproducibility(X, y, centres, methods, task, k='default'):
    """Selection on each centre separately; pairwise agreement between centres."""
    y, centres = np.asarray(y).astype(int), np.asarray(centres)
    usable = [c for c in np.unique(centres)
              if (centres == c).sum() >= config.STABILITY_MIN_CENTRE and y[centres == c].sum() >= 3
              and (y[centres == c] == 0).sum() >= 3]
    rows, per_centre = [], {}
    if len(usable) < 2:
        print(f"  [centres] fewer than 2 usable centres ({usable}); skipped")
        return pd.DataFrame(), {}
    for m in methods:
        if m in FIXED:
            continue
        sets, ranks = {}, {}
        for c in usable:
            msk = centres == c
            r = run_selector(m, X[msk].reset_index(drop=True), y[msk], task, k, seed=0, n_jobs=-1)
            sets[c], ranks[c] = r.selected, r.ranking
        rows.append(_summary(m, list(sets.values()), list(ranks.values()), X.shape[1],
                             f"centres {'/'.join(usable)}"))
        per_centre[m] = sets
        print(f"  [centres]   {m:18s} jaccard={rows[-1]['jaccard']:.3f}")
    return pd.DataFrame(rows), per_centre


def method_agreement(selections):
    """Jaccard between the final selections of every pair of methods."""
    ms = list(selections)
    M = pd.DataFrame(np.eye(len(ms)), index=ms, columns=ms)
    for a, b in combinations(ms, 2):
        M.loc[a, b] = M.loc[b, a] = jaccard(selections[a], selections[b])
    return M
