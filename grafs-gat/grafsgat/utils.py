"""Shared helpers: seeding, metrics, plots."""

import os
import random

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score

from grafsgat import config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(verbose=True):
    dev = torch.device('cuda' if (config.USE_CUDA and torch.cuda.is_available()) else 'cpu')
    if verbose:
        print(f"Using {dev}" + (f": {torch.cuda.get_device_name(0)}" if dev.type == 'cuda' else ''))
    return dev


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_prob))


def find_optimal_threshold(y_true, y_prob):
    """Youden's J on the given data (use the VALIDATION set). Always finite."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    return float(thr[np.argmax(tpr - fpr)]) if len(thr) else 0.5


def calculate_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'auc'        : safe_auc(y_true, y_prob),
        'accuracy'   : float((tp + tn) / max(len(y_true), 1)),
        'sensitivity': float(tp / (tp + fn)) if tp + fn else float('nan'),
        'specificity': float(tn / (tn + fp)) if tn + fp else float('nan'),
        'precision'  : float(tp / (tp + fp)) if tp + fp else float('nan'),
        'f1'         : float(f1_score(y_true, y_pred, zero_division=0)),
        'threshold'  : float(threshold),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'n': int(len(y_true)), 'n_positive': int(y_true.sum()),
    }


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=None, level=None, seed=None):
    n_bootstrap = n_bootstrap or config.N_BOOTSTRAP
    level = level or config.CONFIDENCE_LEVEL
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    aucs = []
    for _ in range(n_bootstrap):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) == 2:
            aucs.append(roc_auc_score(y_true[i], y_prob[i]))
    if not aucs:
        return float('nan'), float('nan')
    a = (1 - level) / 2
    return float(np.percentile(aucs, 100 * a)), float(np.percentile(aucs, 100 * (1 - a)))


def print_metrics(m, title=''):
    print(f"\n{title}")
    print("-" * 55)
    for k in ('auc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'threshold'):
        print(f"  {k:12s}: {m[k]:.4f}")
    if 'auc_ci_low' in m:
        print(f"  AUC 95% CI  : [{m['auc_ci_low']:.4f}, {m['auc_ci_high']:.4f}]")
    print(f"  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}  (n={m['n']}, positives={m['n_positive']})")
    print("-" * 55)


def plot_roc_curve(y_true, y_prob, save_path, title='ROC'):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc_score(y_true, y_prob):.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title(title)
    plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI); plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path, title='Confusion matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Neg', 'Pos']); ax.set_yticklabels(['Neg', 'Pos'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=config.FIGURE_DPI); plt.close()


def plot_training_history(history, save_path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.plot(history['train_loss'], label='train'); a1.plot(history['val_loss'], label='val')
    a1.axvline(history['best_epoch'] - 1, color='grey', ls='--', label='selected')
    a1.set_xlabel('epoch'); a1.set_ylabel('BCE loss'); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(history['val_auc'], color='green', label='val AUC')
    a2.axvline(history['best_epoch'] - 1, color='grey', ls='--')
    a2.set_xlabel('epoch'); a2.set_ylabel('AUC'); a2.legend(); a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
