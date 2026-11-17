import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch

from evst.data import kba
from evst.models import multisdm
from evst.paths import RESULTS
from evst.validate.metrics import auc

ap = argparse.ArgumentParser()
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--budgets", default="0,10,20,40,80,160,320")
ap.add_argument("--steps", type=int, default=300)
a = ap.parse_args()
budgets = [int(x) for x in a.budgets.split(",")]

S = kba.build()
Xz, tg, Ytg, Utg = kba.training_arrays(S)
dev = "cuda"
Xl, tg_t = torch.as_tensor(Xz, device=dev), torch.as_tensor(tg, device=dev)
Yt, Ut = torch.as_tensor(Ytg, device=dev), torch.as_tensor(Utg, device=dev)
Y, R, U, cxy = S["Y"], S["R"], S["U"], S["cxy"]
nC = len(S["ev"])
pix_of = [S["pix_li"][S["pix_k"] == k] for k in range(nC)]
LOSS = "occ_effort"
base = multisdm.fit(LOSS, Xl, tg_t, Yt, Ut, steps=1500, seed=0)
init = copy.deepcopy(base.state_dict())


def pa_tensors(cells):
    pix = np.concatenate([pix_of[c] for c in cells])
    seg = np.concatenate([np.full(len(pix_of[c]), i) for i, c in enumerate(cells)])
    return (torch.as_tensor(pix, device=dev), torch.as_tensor(seg, device=dev), len(cells),
            torch.as_tensor(Y[cells].astype(np.float32), device=dev))


def cell_psi(net):
    return kba.cell_mean(multisdm.score(net, LOSS, Xl), S)


def evaluate(psi, test):
    all_, miss = [], []
    for j in range(Y.shape[1]):
        y = Y[test, j]
        if 5 <= y.sum() <= len(y) - 5:
            all_.append(auc(y, psi[test, j]))
        m = R[test, j] == 0
        ym = y[m]
        if 5 <= ym.sum() <= len(ym) - 5:
            miss.append(auc(ym, psi[test, j][m]))
    return np.mean(all_), np.mean(miss)


def choose(strategy, pool, revealed, k, psi, rng):
    pool = np.array([c for c in pool if c not in revealed])
    if strategy == "random":
        return list(rng.choice(pool, k, replace=False))
    if strategy == "low_effort":
        key = U[pool] + rng.uniform(0, 0.5, len(pool))
        return list(pool[np.argsort(key)[:k]])
    if strategy == "uncertainty":
        key = (psi[pool] * (1 - psi[pool])).sum(1)
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


rows = []
blk = (np.floor(cxy[:, 0] / 0.3) * 1000 + np.floor(cxy[:, 1] / 0.3)).astype(int)
for split in range(a.splits):
    rng = np.random.default_rng(split)
    ub = np.unique(blk)
    test_b = rng.choice(ub, int(0.3 * len(ub)), replace=False)
    test = np.flatnonzero(np.isin(blk, test_b))
    pool = np.setdiff1d(np.arange(nC), test)
    psi0 = cell_psi(base)
    a0, m0 = evaluate(psi0, test)
    for strat in ("random", "low_effort", "uncertainty", "discovery", "spread"):
        t0 = time.time()
        srng = np.random.default_rng(100 + split)
        revealed, psi = [], psi0
        rows.append({"split": split, "strategy": strat, "model": "po+pa", "n_cells": 0, "auc_all": a0, "auc_missing": m0,
                     "test_cells": len(test), "revealed_unvisited_frac": np.nan})
        for nb in budgets[1:]:
            revealed += choose(strat, pool, revealed, nb - len(revealed), psi, srng)
            net = multisdm.fit(LOSS, Xl, tg_t, Yt, Ut, steps=a.steps, seed=split, init=init, pa=pa_tensors(revealed))
            psi = cell_psi(net)
            aa, mm = evaluate(psi, test)
            rows.append({"split": split, "strategy": strat, "model": "po+pa", "n_cells": nb, "auc_all": aa, "auc_missing": mm,
                         "test_cells": len(test), "revealed_unvisited_frac": float((U[revealed] == 0).mean())})
            if strat == "random":
                net2 = multisdm.fit(LOSS, Xl, tg_t, Yt, Ut, steps=600, seed=split, pa=pa_tensors(revealed), po_weight=0.0)
                a2, m2 = evaluate(cell_psi(net2), test)
                rows.append({"split": split, "strategy": strat, "model": "pa_only", "n_cells": nb, "auc_all": a2,
                             "auc_missing": m2, "test_cells": len(test), "revealed_unvisited_frac": np.nan})
        print(split, strat, f"{time.time() - t0:.0f}s", round(aa, 4), round(mm, 4), flush=True)

D = pd.DataFrame(rows)
D.to_csv(RESULTS / "E13_atlas_oracle_allocation.csv", index=False)
print(D.groupby(["model", "strategy", "n_cells"])[["auc_all", "auc_missing"]].mean().round(4).unstack(0).to_string())
