from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans


def random_folds(n: int, k: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % k)


def spatial_folds(x, y, size: float, k: int = 5, seed: int = 0) -> np.ndarray:
    x, y = np.asarray(x, float), np.asarray(y, float)
    bx = np.floor((x - x.min()) / size).astype(np.int64)
    by = np.floor((y - y.min()) / size).astype(np.int64)
    block = bx * 1_000_003 + by
    ub, inv = np.unique(block, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.bincount(inv)
    order = rng.permutation(len(ub))
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_of_block, load = np.empty(len(ub), int), np.zeros(k)
    for b in order:
        f = int(np.argmin(load + rng.random(k) * 1e-9))
        fold_of_block[b] = f
        load[f] += counts[b]
    return fold_of_block[inv]


def env_folds(X, k: int = 5, seed: int = 0) -> np.ndarray:
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return KMeans(k, n_init=10, random_state=seed).fit_predict(Z)


def _expvar(h, nugget, sill, rng):
    return nugget + sill * (1 - np.exp(-3 * h / rng))


def autocorr_range(x, y, values, n_pairs: int = 200_000, n_bins: int = 30, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    v = (values - values.mean()) / (values.std() + 1e-12)
    n = len(v)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    h = np.hypot(x[i] - x[j], y[i] - y[j])
    g = 0.5 * (v[i] - v[j]) ** 2
    hmax = np.quantile(h, 0.5)
    keep = (h > 0) & (h <= hmax)
    edges = np.linspace(0, hmax, n_bins + 1)
    idx = np.digitize(h[keep], edges) - 1
    hb = np.array([h[keep][idx == b].mean() for b in range(n_bins) if (idx == b).sum() > 30])
    gb = np.array([g[keep][idx == b].mean() for b in range(n_bins) if (idx == b).sum() > 30])
    try:
        p, _ = curve_fit(_expvar, hb, gb, p0=[gb.min(), gb.max() - gb.min(), hmax / 3],
                         bounds=([0, 1e-6, hmax / 1000], [np.inf, np.inf, hmax * 10]), maxfev=20000)
        return float(p[2])
    except RuntimeError:
        return float("nan")
