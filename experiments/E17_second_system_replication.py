import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

from evst.data import systems
from evst.models import multisdm
from evst.models.sdm import BRT
from evst.paths import PROCESSED, RESULTS
from evst.validate.blocks import random_folds
from evst.validate.metrics import auc

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--system", default="taiwan")
ap.add_argument("--jobs", type=int, default=10)
a = ap.parse_args()
suf = systems.suffix(a.system)
t0 = time.time()
S = systems.get(a.system)
X, b, cand = S["X"], S["birds"], S["cand"]
Y, R, U = S["Y"], S["R"], S["U"]
tg = np.unique(b.li.to_numpy())
rng = np.random.default_rng(0)
bg = rng.choice(len(X), 20000, replace=False)
bgte = rng.choice(len(X), 5000, replace=False)
cellsets = b.groupby("species").li.unique()
pix_li = S["pix_li"]


def one(sp):
    pc = np.unique(cellsets[sp])
    preds, cv = {}, {}
    for corr, g in (("naive", bg), ("tgb", tg)):
        g = np.setdiff1d(g, pc)
        preds[f"BRT_{corr}"] = BRT().fit(np.r_[X[pc], X[g]], np.r_[np.ones(len(pc)), np.zeros(len(g))]).predict(X[pix_li])
        folds = random_folds(len(pc), 5, 1)
        aucs = []
        for k in range(5):
            mk = BRT().fit(np.r_[X[pc[folds != k]], X[g]], np.r_[np.ones((folds != k).sum()), np.zeros(len(g))])
            s1, s0 = mk.predict(X[pc[folds == k]]), mk.predict(X[bgte])
            aucs.append(auc(np.r_[np.ones(len(s1)), np.zeros(len(s0))], np.r_[s1, s0]))
        cv[corr] = float(np.mean(aucs))
    return preds, cv


out = Parallel(n_jobs=a.jobs)(delayed(one)(sp) for sp in cand)
print("BRT", round(time.time() - t0), "s", flush=True)
sc = {}
for k in ("BRT_naive", "BRT_tgb"):
    pl = np.zeros((len(X), len(cand)), np.float32)
    pl[pix_li] = np.column_stack([o[0][k] for o in out])
    sc[k] = systems.cell_mean(pl, S)
rc = S["rc"].astype(float)
cc = systems.cell_mean(rc, S)
sc["nearest_record"] = np.column_stack([-cKDTree(rc[np.unique(cellsets[sp])]).query(cc)[0] for sp in cand])
Xz, tgn, Ytg, Utg = systems.training_arrays(S)
T = [torch.as_tensor(v, device="cuda") for v in (Xz, tgn, Ytg, Utg)]
for loss in ("an_ssdl", "occ_effort"):
    acc = 0
    for seed in range(5):
        net = multisdm.fit(loss, *T, seed=seed)
        p = multisdm.score(net, loss, T[0])
        acc = acc + p / p.mean(0, keepdims=True)
    sc[f"NN_{loss}"] = systems.cell_mean(acc / 5, S)
np.savez_compressed(PROCESSED / f"E17_scores{suf}.npz", **sc)
print("NN", round(time.time() - t0), "s", flush=True)

rows, hits = [], []
for j, sp in enumerate(cand):
    m = R[:, j] == 0
    y = Y[m, j]
    if 5 <= y.sum() <= len(y) - 5:
        rows.append({"species": sp, "endemic": bool(S["endemic"][j]), "n": int(m.sum()), "prev": y.mean(),
                     **{k: auc(y, v[m, j]) for k, v in sc.items()}})
D = pd.DataFrame(rows)
D.to_csv(RESULTS / f"E17_missing_pair_auc{suf}.csv", index=False)
models = list(sc)
mask = R == 0
for k, v in sc.items():
    rk = np.full(v.shape, np.nan)
    for j in range(len(cand)):
        mm = mask[:, j]
        if mm.sum():
            r = v[mm, j].argsort().argsort() / max(mm.sum() - 1, 1)
            rk[mm, j] = r
    top = (rk >= 0.9) & mask
    hits.append({"model": k, "top10_hit_rate": Y[top].mean(), "base_rate": Y[mask].mean()})
H = pd.DataFrame(hits)
cs = []
for x, ref in (("BRT_tgb", "BRT_naive"), ("BRT_tgb", "nearest_record"), ("NN_occ_effort", "NN_an_ssdl"),
               ("NN_occ_effort", "BRT_tgb")):
    for grp, g in (("all", D), ("endemic", D[D.endemic]), ("non_endemic", D[~D.endemic])):
        if len(g) >= 5:
            d = g[x] - g[ref]
            cs.append({"group": grp, "contrast": f"{x} - {ref}", "n_species": len(g), "delta": d.mean(),
                       "pct_better": (d > 0).mean(), "wilcoxon_p": wilcoxon(d).pvalue})
C = pd.DataFrame(cs)
C.to_csv(RESULTS / f"E17_contrasts{suf}.csv", index=False)
cvr = []
for j, sp in enumerate(cand):
    y = Y[:, j]
    if 5 <= y.sum() <= len(y) - 5:
        cv = out[j][1]
        cvr.append({"species": sp, "cv_naive": cv["naive"], "cv_tgb": cv["tgb"], "truth_naive": auc(y, sc["BRT_naive"][:, j]),
                    "truth_tgb": auc(y, sc["BRT_tgb"][:, j])})
V = pd.DataFrame(cvr)
V.to_csv(RESULTS / f"E17_cv_vs_truth{suf}.csv", index=False)
dcv, dtr = V.cv_tgb - V.cv_naive, V.truth_tgb - V.truth_naive
inv = {"cv_prefers_tgb": (dcv > 0).mean(), "truth_prefers_tgb": (dtr > 0).mean(),
       "agreement": (np.sign(dcv) == np.sign(dtr)).mean(), "optimism_naive": (V.cv_naive - V.truth_naive).mean(),
       "optimism_tgb": (V.cv_tgb - V.truth_tgb).mean(), "n_species": len(V)}
pd.DataFrame([inv]).to_csv(RESULTS / f"E17_cv_inversion{suf}.csv", index=False)
print(f"{len(S['ev'])} sites, {len(cand)} species, {mask.sum()} unrecorded pairs, survey found {Y[mask].mean():.3f}")
print(D[models].mean().round(4).to_string())
print(H.round(4).to_string(index=False))
print(C.round(4).to_string(index=False))
print({k: round(v, 4) for k, v in inv.items()})
print("total", round(time.time() - t0), "s")
