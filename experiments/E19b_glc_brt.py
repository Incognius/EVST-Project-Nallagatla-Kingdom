"""E19b: classical reference for E19 -- per-species BRT (LightGBM, project settings) on the 1 km centre pixel,
random vs target-group background, same 80% random training split of PO cells as E19 (seed 0).
A random subset of species keeps the CPU cost bounded. Same outputs as E19 (PA AUC, access excess)."""
import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from evst.models.sdm import BRT
from evst.paths import INTERIM, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--n-species", type=int, default=300)
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--n-bg", type=int, default=20000)
a = ap.parse_args()
G = INTERIM / "glc24"
meta = json.loads((G / "meta.json").read_text())
H, W = meta["shape"]
stack = np.load(G / "stack.npy", mmap_mode="r")
land = np.load(G / "land_cells.npy")
po, pa = np.load(G / "po_cells.npz"), np.load(G / "pa.npz")
species = np.load(G / "species.npy")
S = len(species)

rng = np.random.default_rng(0)
cells_all = po["cells_all"]
hold = rng.random(len(cells_all)) < 0.2  # identical to E19 --split random --seed 0
train_cells = cells_all[~hold]
rec = pd.DataFrame({"cell": po["cell"], "sp": po["sp"]})
rec = rec[rec.cell.isin(train_cells)].drop_duplicates()
by_sp = rec.groupby("sp").cell.apply(np.array)


def feats(cells):
    cells = np.asarray(cells)
    return np.stack([np.asarray(stack[c].ravel()[cells], np.float32) for c in range(stack.shape[0])], 1)


rs = np.random.default_rng(1)
Ypa = pa["Y"]
eligible = np.flatnonzero(Ypa.sum(0) >= 10)
chosen = np.sort(rs.choice(eligible, min(a.n_species, len(eligible)), replace=False))
bg = {"random": rs.choice(land, a.n_bg, replace=False), "tgb": rs.choice(train_cells, a.n_bg, replace=False)}
Xbg = {k: feats(v) for k, v in bg.items()}
Xpa = feats(pa["cell"])
Az = pa["access"]
names = meta["access_names"] + ["access_index"]


def auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def one(s):
    pres = by_sp.get(s, np.array([], np.int64))
    if len(pres) > 5000:
        pres = np.random.default_rng(s).choice(pres, 5000, replace=False)
    Xp = feats(pres)
    out = {"species": species[s], "n_train_pres": len(pres)}
    y = Ypa[:, s].astype(int)
    for k in ("random", "tgb"):
        keep = ~np.isin(bg[k], pres)
        m = BRT().fit(np.r_[Xp, Xbg[k][keep]], np.r_[np.ones(len(Xp)), np.zeros(keep.sum())])
        p = m.predict(Xpa)
        out[f"auc_pa_{k}"] = auc(y, p)
        top = np.argsort(-p)[:y.sum()]
        for j, nm in enumerate(names):
            a_j = Az[:, j] if j < Az.shape[1] else Az.mean(1)
            out[f"excess_{nm}_{k}"] = a_j[top].mean() - a_j[y == 1].mean()
    return out


R = pd.DataFrame(Parallel(n_jobs=a.jobs, verbose=5)(delayed(one)(s) for s in chosen))
(RESULTS / "E19").mkdir(exist_ok=True)
R.to_csv(RESULTS / "E19" / "E19b_brt.csv", index=False)
print(R[["auc_pa_random", "auc_pa_tgb", "excess_access_index_random", "excess_access_index_tgb"]].mean().round(4))
