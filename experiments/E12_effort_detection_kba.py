import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon

from evst.data import kba
from evst.models import multisdm
from evst.paths import PROCESSED, RESULTS
from evst.validate.metrics import auc

S = kba.build()
Xz, tg, Ytg, Utg = kba.training_arrays(S)
dev = "cuda"
Xl, tg_t = torch.as_tensor(Xz, device=dev), torch.as_tensor(tg, device=dev)
Yt, Ut = torch.as_tensor(Ytg, device=dev), torch.as_tensor(Utg, device=dev)
Y, R, U = S["Y"], S["R"], S["U"]
cand = S["cand"]

fits = {}
for name, bal in (("occ_effort", True), ("occ_effort_unweighted", False)):
    psi, det = 0, 0
    for seed in range(5):
        net = multisdm.fit("occ_effort", Xl, tg_t, Yt, Ut, steps=1500, seed=seed, balanced=bal)
        psi = psi + multisdm.score(net, "occ_effort", Xl)
        det = det + torch.sigmoid(net.det).detach().cpu().numpy()
    fits[name] = (kba.cell_mean(psi / 5, S), det / 5)
    np.save(PROCESSED / f"E12_cell_psi_{name}.npy", fits[name][0])
    np.save(PROCESSED / f"E12_det_{name}.npy", fits[name][1])
    print(name, "median per-visit detection", np.round(np.median(det / 5), 4), flush=True)

scores = {k: np.load(PROCESSED / f"E11_cell_scores_{k}.npy") for k in ("an_slds", "an_ssdl", "maxent_tgb")}
scores["occ_effort"] = fits["occ_effort"][0]
scores["occ_effort_unweighted"] = fits["occ_effort_unweighted"][0]
P = pd.read_csv(RESULTS / "E09b_kba_unrecorded_pairs.csv.gz", usecols=["species", "kcell", "BRT_tgb"])
kpos = pd.Series(np.arange(len(S["ev"])), index=S["ev"])
spos = pd.Series(np.arange(len(cand)), index=cand)
P = P[P.kcell.isin(kpos.index) & P.species.isin(spos.index)]
brt = np.full(Y.shape, np.nan)
brt[kpos[P.kcell].to_numpy(), spos[P.species].to_numpy()] = P.BRT_tgb.to_numpy()
scores["BRT_tgb"] = brt

strata = {"no_inat_visits": U == 0, "visits_1to5": (U >= 1) & (U <= 5), "visits_gt5": U > 5}
rows = []
for sname, cm in strata.items():
    for j, sp in enumerate(cand):
        m = cm & (R[:, j] == 0)
        y = Y[m, j]
        if y.sum() < 5 or y.sum() > len(y) - 5:
            continue
        rows.append({"stratum": sname, "species": sp, "n_cells": int(m.sum()),
                     **{k: (np.nan if np.isnan(v[m, j]).any() else auc(y, v[m, j])) for k, v in scores.items()}})
T = pd.DataFrame(rows)
T.to_csv(RESULTS / "E12_effort_strata_auc.csv", index=False)
out = []
for sname, g in T.groupby("stratum"):
    g = g.dropna()
    for k in ("occ_effort", "occ_effort_unweighted", "maxent_tgb", "BRT_tgb"):
        d = g[k] - g["an_slds"]
        out.append({"stratum": sname, "model": k, "n_species": len(g), "mean_auc": g[k].mean(), "an_slds_auc": g.an_slds.mean(),
                    "delta_vs_an_slds": d.mean(), "pct_better": (d > 0).mean(), "wilcoxon_p": wilcoxon(d).pvalue})
O = pd.DataFrame(out)
print(O.round(4).to_string(index=False))

cal = []
E9 = pd.read_csv(RESULTS / "E09_kba_cells.csv").set_index("kcell").reindex(S["ev"])
for name in ("occ_effort", "occ_effort_unweighted"):
    p = fits[name][1]
    pred = 1 - (1 - p[None, :]) ** U[:, None]
    sampled = U > 0
    obs_cell = np.array([(R[i] * Y[i]).sum() / max(Y[i].sum(), 1) for i in range(len(U))])
    pred_cell = np.array([(pred[i] * Y[i]).sum() / max(Y[i].sum(), 1) for i in range(len(U))])
    for lo, hi in ((1, 1), (2, 5), (6, 20), (21, 10000)):
        m = (U >= lo) & (U <= hi)
        cal.append({"model": name, "visits": f"{lo}-{hi}" if hi < 10000 else f">{lo - 1}", "n_cells": int(m.sum()),
                    "observed_recorded_frac": obs_cell[m].mean(), "predicted_recorded_frac": pred_cell[m].mean(),
                    "bayes_discovery_pred": 1 - E9.bayes_full_P0.values[m].mean()})
    print(name, "spearman obs vs pred recorded fraction (sampled cells):",
          round(spearmanr(obs_cell[sampled], pred_cell[sampled])[0], 3),
          "| MAE", round(np.abs(obs_cell[sampled] - pred_cell[sampled]).mean(), 3),
          "| bayes MAE", round(np.abs(obs_cell[sampled] - (1 - E9.bayes_full_P0.values[sampled])).mean(), 3))
C = pd.DataFrame(cal)
print(C.round(3).to_string(index=False))
O.to_csv(RESULTS / "E12_effort_strata_summary.csv", index=False)
C.to_csv(RESULTS / "E12_detection_calibration.csv", index=False)
