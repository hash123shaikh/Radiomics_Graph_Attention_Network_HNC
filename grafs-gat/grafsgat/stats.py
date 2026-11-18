"""
Statistics for comparing models on the same patients.

  delong_test(y, p1, p2)  paired DeLong test for two correlated AUCs (Sun & Xu 2014 fast version)
  holm(pvalues)           Holm-Bonferroni adjusted p-values
"""

import numpy as np
from scipy.stats import norm


def _midrank(x):
    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    n = len(x)
    T = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = T
    return out


def _fast_delong(preds_sorted, m):
    """preds_sorted: (k, n) with the m positives first."""
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx, sy = np.atleast_2d(np.cov(v01)), np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y, p1, p2):
    """Returns (auc1, auc2, z, two-sided p) for paired predictions on the same patients."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind='stable')
    m = int(y.sum())
    if m == 0 or m == len(y):
        return np.nan, np.nan, np.nan, np.nan
    P = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs, cov = _fast_delong(P, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * norm.sf(abs(z)))


def delong_ci(y, p, level=0.95):
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind='stable')
    m = int(y.sum())
    if m == 0 or m == len(y):
        return np.nan, np.nan
    aucs, cov = _fast_delong(np.asarray(p)[order][None, :], m)
    se = np.sqrt(cov[0, 0])
    zc = norm.ppf(1 - (1 - level) / 2)
    return float(max(0, aucs[0] - zc * se)), float(min(1, aucs[0] + zc * se))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m, running = len(idx), 0.0
    for r, i in enumerate(idx):
        running = max(running, min(1.0, (m - r) * p[i]))
        out[i] = running
    return out
