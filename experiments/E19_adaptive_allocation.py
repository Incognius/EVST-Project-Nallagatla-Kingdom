import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch

from evst.allocate import acquire
from evst.data import systems
from evst.models import multisdm
from evst.paths import RESULTS
from evst.validate.metrics import auc

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="kba")
ap.add_argument("--splits", type=int, default=10)
ap.add_argument("--budgets", default="0,10,20,40,80,160,320")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--strategies", default=",".join(acquire.BASE + acquire.NEW + acquire.ORACLE))
a = ap.parse_args()
budgets = [int(x) for x in a.budgets.split(",")]
strategies = a.strategies.split(",")
suf = systems.suffix(a.system)

S = systems.get(a.system)
Xz, tg, Ytg, Utg = systems.training_arrays(S)
dev = "cuda"
Xl, tg_t = torch.as_tensor(Xz, device=dev), torch.as_tensor(tg, device=dev)
Yt, Ut = torch.as_tensor(Ytg, device=dev), torch.as_tensor(Utg, device=dev)
Y, R, U, cxy = S["Y"], S["R"], S["U"], S["cxy"]
nC = len(S["ev"])
budgets = [b for b in budgets if b <= 0.7 * nC * 0.9]
pix_of = [S["pix_li"][S["pix_k"] == k] for k in range(nC)]
LOSS = "occ_effort"
base = multisdm.fit(LOSS, Xl, tg_t, Yt, Ut, steps=1500, seed=0)
init = copy.deepcopy(base.state_dict())
Xc = systems.cell_mean(Xz, S)
Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
u_, sv, _ = np.linalg.svd(Xc, full_matrices=False)
Ze = u_[:, :5] * sv[:5]
feat = np.column_stack([(Ze - Ze.mean(0)) / Ze.std(0), (cxy - cxy.mean(0)) / cxy.std(0)])
pix_t = torch.as_tensor(S["pix_li"], device=dev)
cnt = np.bincount(S["pix_k"], minlength=nC)[:, None]


def pa_tensors(cells):
    pix = np.concatenate([pix_of[c] for c in cells])
    seg = np.concatenate([np.full(len(pix_of[c]), i) for i, c in enumerate(cells)])
    return (torch.as_tensor(pix, device=dev), torch.as_tensor(seg, device=dev), len(cells),
            torch.as_tensor(Y[cells].astype(np.float32), device=dev))


def cell_psi(net):
    return systems.cell_mean(multisdm.score(net, LOSS, Xl), S)


def sampler(net):
    def f():
        net.train()
        with torch.no_grad():
            p = torch.sigmoid(net(Xl[pix_t])).cpu().numpy()
        out = np.zeros((nC, p.shape[1]))
        np.add.at(out, S["pix_k"], p)
        return out / cnt
    return f


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


rows = []
blk = (np.floor(cxy[:, 0] / 0.3) * 1000 + np.floor(cxy[:, 1] / 0.3)).astype(int)
psi0 = cell_psi(base)
for split in range(a.splits):
    rng = np.random.default_rng(split)
    ub = np.unique(blk)
    test_b = rng.choice(ub, int(0.3 * len(ub)), replace=False)
    test = np.flatnonzero(np.isin(blk, test_b))
    pool = np.setdiff1d(np.arange(nC), test)
    a0, m0 = evaluate(psi0, test)
    for strat in strategies:
        t0 = time.time()
        srng = np.random.default_rng(100 + split)
        revealed, psi, net = [], psi0, base
        ctx = {"R": R, "U": U, "cxy": cxy, "feat": feat, "Y": Y, "min_sep": 0.15}
        rows.append({"system": a.system, "split": split, "strategy": strat, "model": "po+pa", "n_cells": 0, "auc_all": a0,
                     "auc_missing": m0, "test_cells": len(test), "revealed_unvisited_frac": np.nan, "discovery_yield": np.nan})
        for nb in budgets[1:]:
            ctx.update(psi=psi, sample=sampler(net))
            revealed += acquire.choose(strat, pool, revealed, nb - len(revealed), ctx, srng)
            net = multisdm.fit(LOSS, Xl, tg_t, Yt, Ut, steps=a.steps, seed=split, init=init, pa=pa_tensors(revealed))
            new = cell_psi(net)
            if strat == "adaptive":
                acquire.update_lambda(ctx, new[pool], psi[pool])
            psi = new
            aa, mm = evaluate(psi, test)
            rv = np.array(revealed)
            rows.append({"system": a.system, "split": split, "strategy": strat, "model": "po+pa", "n_cells": nb,
                         "auc_all": aa, "auc_missing": mm, "test_cells": len(test),
                         "revealed_unvisited_frac": float((U[rv] == 0).mean()),
                         "discovery_yield": float((Y[rv] * (1 - R[rv])).sum(1).mean())})
        print(split, strat, f"{time.time() - t0:.0f}s", round(aa, 4), round(mm, 4), flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"E19_adaptive_allocation{suf}.csv", index=False)

D = pd.DataFrame(rows)
print(D.groupby(["strategy", "n_cells"])[["auc_all", "discovery_yield", "revealed_unvisited_frac"]].mean().round(4)
      .unstack(0)["auc_all"].to_string())
