from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


def auc(y, s) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, s))


def cor(y, s) -> float:
    y = np.asarray(y, dtype=float)
    if y.min() == y.max() or np.std(s) == 0:
        return np.nan
    return float(pearsonr(y, s)[0])


def boyce(pred_pres, pred_bg, n_bins: int = 101, window: float = 0.1) -> float:
    lo, hi = min(pred_bg.min(), pred_pres.min()), max(pred_bg.max(), pred_pres.max())
    w = (hi - lo) * window
    starts = np.linspace(lo, hi - w, n_bins)
    pe = []
    for s in starts:
        p = ((pred_pres >= s) & (pred_pres <= s + w)).mean()
        e = ((pred_bg >= s) & (pred_bg <= s + w)).mean()
        pe.append(p / e if e > 0 else np.nan)
    pe = np.array(pe)
    ok = ~np.isnan(pe)
    return float(spearmanr(starts[ok] + w / 2, pe[ok])[0]) if ok.sum() > 2 else np.nan


def leakage(pred, bias_layers, truth=None) -> float:
    from numpy.linalg import lstsq
    p = np.clip(np.asarray(pred, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    B = np.column_stack([np.ones(len(z)), np.asarray(bias_layers, float).reshape(len(z), -1)])

    def rss(M):
        beta = lstsq(M, z, rcond=None)[0]
        return float(((z - M @ beta) ** 2).sum())

    if truth is None:
        return 1 - rss(B) / float(((z - z.mean()) ** 2).sum())
    T = np.column_stack([np.ones(len(z)), np.asarray(truth, float).reshape(len(z), -1)])
    r_t = rss(T)
    r_tb = rss(np.column_stack([T, B[:, 1:]]))
    return 1 - r_tb / r_t
