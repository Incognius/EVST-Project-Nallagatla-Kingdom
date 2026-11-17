import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.stats import kendalltau, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

from evst.data import kba
from evst.models.sdm import BRT
from evst.paths import PROCESSED, RESULTS
from evst.truth import occupancy as oc

warnings.filterwarnings("ignore")
t0 = time.time()
S = kba.build()
cand = S["cand"]
Z = oc.env_pcs(S)


def occupancy(St, tag):
    H = oc.histories(St)
    fits = {s: oc.fit(Z, H[s]) for s in oc.SEASONS}
    gate = []
    for s in oc.SEASONS:
        Hs, z = oc.simulate(fits[s], H[s])
        r2 = oc.fit(Z, Hs)
        e = np.abs(r2["psi"].mean(0) - fits[s]["psi"].mean(0))
        und = r2["det"] == 0
        gate.append({"set": tag, "season": s, "psi_err_median": np.median(e), "psi_err_p90": np.quantile(e, 0.9),
                     "pfa_pred_mean": r2["pfa"][und].mean(), "pfa_true_frac": z[und].mean(),
                     "median_p_list_2obs": np.median(fits[s]["p_two_obs"]), "mean_psi": fits[s]["psi"].mean(),
                     "naive_occ": fits[s]["det"].mean()})
    return fits, oc.union_pfa(fits), gate


fits, u, gate = occupancy(S, "full_2020")
np.savez_compressed(PROCESSED / "E16_atlas_occupancy.npz", pfa=u, **{f"psi_{s}": fits[s]["psi"] for s in oc.SEASONS},
                    **{f"det_{s}": fits[s]["det"] for s in oc.SEASONS}, **{f"p2_{s}": fits[s]["p_two_obs"] for s in oc.SEASONS})
print("occupancy", round(time.time() - t0), "s", flush=True)

cache = PROCESSED / "E16_brt_scores.npz"
if cache.exists():
    sc = dict(np.load(cache))
else:
    X = S["X"]
    b = S["birds"]
    tg = np.unique(b.li.to_numpy())
    bg = np.random.default_rng(0).choice(len(X), 20000, replace=False)
    cellsets = b.groupby("species").li.unique()
    pix_li = S["pix_li"]

    def brt(sp):
        pc = np.unique(cellsets[sp])
        return [BRT().fit(np.r_[X[pc], X[np.setdiff1d(g, pc)]], np.r_[np.ones(len(pc)), np.zeros(len(np.setdiff1d(g, pc)))])
                .predict(X[pix_li]) for g in (bg, tg)]

    res = Parallel(n_jobs=10)(delayed(brt)(sp) for sp in cand)
    sc = {}
    for i, k in enumerate(("BRT_naive", "BRT_tgb")):
        pl = np.zeros((len(X), len(cand)), np.float32)
        pl[pix_li] = np.column_stack([r[i] for r in res])
        sc[k] = kba.cell_mean(pl, S)
    rc = S["rc"].astype(float)
    cc = kba.cell_mean(rc, S)
    nr = np.zeros((len(S["ev"]), len(cand)))
    for j, sp in enumerate(cand):
        nr[:, j] = -cKDTree(rc[np.unique(cellsets[sp])]).query(cc)[0]
    sc["nearest_record"] = nr
    np.savez_compressed(cache, **sc)
print("BRT scores", round(time.time() - t0), "s", flush=True)
for loss in ("an_ssdl", "an_slds", "an_full", "maxent_random", "maxent_tgb", "occ_effort"):
    sc[loss] = np.load(PROCESSED / f"E11_cell_scores_{loss}.npy")
models = list(sc)

Y, R = S["Y"], S["R"]
dW, dD = fits["Wet"]["det"], fits["Dry"]["det"]


def soft_auc(y, w_pos, s):
    yy = np.r_[np.ones(len(y)), np.zeros(len(y))]
    ww = np.r_[w_pos, 1 - w_pos]
    return roc_auc_score(yy, np.r_[s, s], sample_weight=ww)


rows = []
for j, sp in enumerate(cand):
    m = R[:, j] == 0
    y, uj = Y[m, j], u[m, j]
    if not 5 <= y.sum() <= len(y) - 5:
        continue
    keep = ~((y == 0) & (uj > 0.1))
    wpos = np.where(y == 1, 1.0, uj)
    for var, mk, yv in (("atlas", np.ones_like(y, bool), y), ("drop_pfa_gt_0.1", keep, y), ("wet_only", None, dW[m, j]),
                        ("dry_only", None, dD[m, j])):
        mk = np.ones_like(y, bool) if mk is None else mk
        yv = yv[mk]
        if not 5 <= yv.sum() <= len(yv) - 5:
            continue
        rows.append({"species": sp, "variant": var, "n": int(mk.sum()),
                     **{k: roc_auc_score(yv, sc[k][m, j][mk]) for k in models}})
    rows.append({"species": sp, "variant": "soft_occupancy", "n": int(m.sum()),
                 **{k: soft_auc(y, wpos, sc[k][m, j]) for k in models}})
D = pd.DataFrame(rows)
D.to_csv(RESULTS / "E16_rescored_species_auc.csv", index=False)
contr = [("BRT_tgb", "BRT_naive"), ("BRT_tgb", "nearest_record"), ("occ_effort", "an_ssdl"), ("an_slds", "an_ssdl"),
         ("maxent_tgb", "maxent_random"), ("occ_effort", "BRT_tgb"), ("maxent_tgb", "BRT_tgb")]
summ, means = [], []
for var, g in D.groupby("variant"):
    means.append({"variant": var, "n_species": len(g), **g[models].mean().to_dict()})
    for x, ref in contr:
        d = g[x] - g[ref]
        summ.append({"variant": var, "contrast": f"{x} - {ref}", "n_species": len(g), "delta": d.mean(),
                     "pct_better": (d > 0).mean(), "wilcoxon_p": wilcoxon(d).pvalue})
M = pd.DataFrame(means).set_index("variant")
M.to_csv(RESULTS / "E16_rescored_means.csv")
Sm = pd.DataFrame(summ)
Sm.to_csv(RESULTS / "E16_rescored_contrasts.csv", index=False)
rk = [{"a": a, "b": b, "kendall_tau": kendalltau(M.loc[a], M.loc[b])[0]}
      for a, b in (("atlas", "drop_pfa_gt_0.1"), ("atlas", "soft_occupancy"), ("atlas", "wet_only"), ("atlas", "dry_only"),
                   ("wet_only", "dry_only"))]
pd.DataFrame(rk).to_csv(RESULTS / "E16_rank_agreement.csv", index=False)
print(M.round(4).T.to_string())
print(Sm.round(4).to_string(index=False))
print(pd.DataFrame(rk).round(3).to_string(index=False))

S14 = kba.build(end=2014, min_cells=10)
f14, u14, g14 = occupancy(S14, "train_2014")
gate += g14
E = np.load(PROCESSED / "E15_scores_2014.npz")
later, Ul = E["later"], E["Ul"]
keys = ["BRT_naive", "BRT_tgb", "NN_an_ssdl", "NN_occ_effort"]
fl = []
for j, sp in enumerate(S14["cand"]):
    eff = (S14["R"][:, j] == 0) & (Ul > 0)
    y, uj, yl = S14["Y"][eff, j], u14[eff, j], later[eff, j]
    keep = ~((y == 0) & (uj > 0.1))
    for var, yy, mk in (("atlas", y, None), ("drop_pfa_gt_0.1", y, keep), ("later_inat", yl, None)):
        mk = np.ones_like(y, bool) if mk is None else mk
        if 5 <= yy[mk].sum() <= mk.sum() - 5:
            fl.append({"species": sp, "variant": var, **{k: roc_auc_score(yy[mk], E[k][eff, j][mk]) for k in keys}})
    if 5 <= y.sum() <= len(y) - 5:
        fl.append({"species": sp, "variant": "soft_occupancy",
                   **{k: soft_auc(y, np.where(y == 1, 1.0, uj), E[k][eff, j]) for k in keys}})
F = pd.DataFrame(fl)
fs = []
for var, g in F.groupby("variant"):
    for x, ref in (("BRT_tgb", "BRT_naive"), ("NN_occ_effort", "NN_an_ssdl")):
        d = g[x] - g[ref]
        fs.append({"variant": var, "contrast": f"{x} - {ref}", "n_species": len(g), "delta": d.mean(),
                   "pct_better": (d > 0).mean(), "wilcoxon_p": wilcoxon(d).pvalue})
pd.DataFrame(fs).to_csv(RESULTS / "E16_flip_rescored.csv", index=False)
print(pd.DataFrame(fs).round(4).to_string(index=False))

p_atlas = np.mean([fits[s]["p_two_obs"] for s in oc.SEASONS], 0)
det_rows = []
for name in ("occ_effort", "occ_effort_unweighted"):
    pv = np.load(PROCESSED / f"E12_det_{name}.npy")
    det_rows.append({"model": name, "median_inat_visit_p": np.median(pv), "median_atlas_list_p": np.median(p_atlas),
                     "spearman_species": spearmanr(pv, p_atlas)[0], "ratio_median": np.median(pv) / np.median(p_atlas)})
pd.DataFrame(det_rows).to_csv(RESULTS / "E16_detection_compare.csv", index=False)
pd.DataFrame(gate).to_csv(RESULTS / "E16_occupancy_gate.csv", index=False)
print(pd.DataFrame(det_rows).round(4).to_string(index=False))
print(pd.DataFrame(gate).round(4).to_string(index=False))
print("total", round(time.time() - t0), "s")
