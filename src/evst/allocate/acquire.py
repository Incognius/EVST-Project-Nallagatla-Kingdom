from __future__ import annotations

import numpy as np

BASE = ("random", "low_effort", "uncertainty", "discovery", "spread")
NEW = ("effort_disc_unc", "coverage_unc", "adaptive", "thompson")
ORACLE = ("oracle",)


def _unc(psi):
    return (psi * (1 - psi)).sum(1)


def _greedy(pool, covered, k, feat, gain, lam):
    out = []
    pool = pool.copy()
    cov = list(covered)
    for _ in range(k):
        if cov:
            d = np.min(np.linalg.norm(feat[pool][:, None] - feat[cov][None], axis=2), axis=1)
            d = d / (d.max() + 1e-9)
        else:
            d = np.zeros(len(pool))
        g = gain[pool] / (gain[pool].max() + 1e-9)
        c = pool[np.argmax(g + lam * d)]
        out.append(c)
        cov.append(c)
        pool = pool[pool != c]
    return out


def choose(strategy, pool, revealed, k, ctx, rng):
    pool = np.array([c for c in pool if c not in revealed])
    psi, R, U, cxy = ctx["psi"], ctx["R"], ctx["U"], ctx["cxy"]
    if strategy == "random":
        return list(rng.choice(pool, k, replace=False))
    if strategy == "low_effort":
        key = U[pool] + rng.uniform(0, 0.5, len(pool))
        return list(pool[np.argsort(key)[:k]])
    if strategy == "uncertainty":
        key = _unc(psi[pool])
        return list(pool[np.argsort(-key)[:k]])
    if strategy == "discovery":
        key = (psi[pool] * (1 - R[pool])).sum(1)
        return list(pool[np.argsort(-key)[:k]])
    if strategy == "spread":
        chosen = list(revealed)
        out = []
        for _ in range(k):
            if chosen:
                d = np.min(np.linalg.norm(cxy[pool][:, None] - cxy[chosen][None], axis=2), axis=1)
            else:
                d = rng.uniform(size=len(pool))
            c = pool[np.argmax(d)]
            out.append(c)
            chosen.append(c)
            pool = pool[pool != c]
        return out
    if strategy == "effort_disc_unc":
        key = _unc(psi[pool]) / (1 + U[pool])
        return list(pool[np.argsort(-key)[:k]])
    if strategy in ("coverage_unc", "adaptive"):
        covered = list(revealed) + list(np.flatnonzero(U > 0))
        lam = ctx.get("lam", 1.0) if strategy == "adaptive" else 1.0
        return _greedy(pool, covered, k, ctx["feat"], _unc(psi), lam)
    if strategy == "thompson":
        out, picked = [], []
        for _ in range(k):
            ps = ctx["sample"]()
            key = (ps[pool] * (1 - R[pool])).sum(1)
            if picked:
                d = np.min(np.linalg.norm(cxy[pool][:, None] - cxy[picked][None], axis=2), axis=1)
                key = np.where(d < ctx["min_sep"], -np.inf, key)
            c = pool[np.argmax(key)]
            out.append(c)
            picked.append(c)
            pool = pool[pool != c]
        return out
    if strategy == "oracle":
        err = np.abs(ctx["Y"][pool] - psi[pool]).sum(1)
        return list(pool[np.argsort(-err)[:k]])
    raise KeyError(strategy)


def update_lambda(ctx, psi_new, psi_old, lam0=2.0):
    d = float(np.abs(psi_new - psi_old).mean())
    ctx.setdefault("d1", d)
    ctx["lam"] = lam0 * float(np.clip(d / (ctx["d1"] + 1e-12), 0.1, 1.0))
    return ctx["lam"]
