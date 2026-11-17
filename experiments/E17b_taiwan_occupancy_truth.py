import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from evst.data import systems, taiwan
from evst.paths import PROCESSED, RESULTS
from evst.truth import occupancy as oc

S = systems.get("taiwan")
pv = taiwan.point_visits()
pv = pv[pv.year.between(*taiwan.TRUTH_YEARS) & pv.site.isin(S["ev"])].copy()
nm = taiwan.name_map(set(pv.species.explode().dropna()))
pv["k"] = pv.groupby("site").cumcount()
kmax = int(pv.k.max()) + 1
n, m = len(S["ev"]), len(S["cand"])
cpos = pd.Series(np.arange(n), index=S["ev"])
spos = pd.Series(np.arange(m), index=S["cand"])
y = np.zeros((n, kmax, m), np.float32)
mask = np.zeros((n, kmax), np.float32)
morn = np.zeros((n, kmax), np.float32)
ci, ki = cpos[pv.site].to_numpy(), pv.k.to_numpy()
mask[ci, ki] = 1
morn[ci, ki] = (pv.hour.fillna(7) < 7).astype(np.float32).to_numpy()
ex = pv[["site", "k", "species"]].explode("species").dropna()
ex["species"] = ex.species.map(nm)
ex = ex[ex.species.isin(spos.index)]
y[cpos[ex.site].to_numpy(), ex.k.to_numpy(), spos[ex.species].to_numpy()] = 1
h = dict(y=y, mask=mask, obs=np.zeros_like(mask), morn=morn)
det = (y * mask[..., None]).max(1)
assert (det == S["Y"]).all()
Xc = systems.cell_mean(S["X"], S)
Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
u, sv, _ = np.linalg.svd(Xc, full_matrices=False)
Z = (u[:, :5] * sv[:5]).astype(np.float32)
Z = (Z - Z.mean(0)) / Z.std(0)
fit = oc.fit(Z, h)
Hs, z = oc.simulate(fit, h)
r2 = oc.fit(Z, Hs)
e = np.abs(r2["psi"].mean(0) - fit["psi"].mean(0))
und = r2["det"] == 0
gate = {"psi_err_median": np.median(e), "psi_err_p90": np.quantile(e, 0.9), "pfa_pred_mean": r2["pfa"][und].mean(),
        "pfa_true_frac": z[und].mean(), "median_p_point_count": np.median(fit["p_two_obs"]),
        "mean_replicates": mask.sum(1).mean(), "survey_absent_pairs": int((S["Y"] == 0).sum()),
        "mean_pfa_absent": fit["pfa"][S["Y"] == 0].mean(), "frac_pfa_gt_0.1": (fit["pfa"][S["Y"] == 0] > 0.1).mean()}
print({k: round(float(v), 4) for k, v in gate.items()})
sc = dict(np.load(PROCESSED / "E17_scores_taiwan.npz"))
pfa = fit["pfa"]
rows = []
for j, sp in enumerate(S["cand"]):
    mm = S["R"][:, j] == 0
    yy, uj = S["Y"][mm, j], pfa[mm, j]
    keep = ~((yy == 0) & (uj > 0.1))
    for var, msk in (("survey", np.ones_like(yy, bool)), ("drop_pfa_gt_0.1", keep)):
        if 5 <= yy[msk].sum() <= msk.sum() - 5:
            rows.append({"species": sp, "variant": var, **{k: roc_auc_score(yy[msk], v[mm, j][msk]) for k, v in sc.items()}})
D = pd.DataFrame(rows)
out = []
for var, g in D.groupby("variant"):
    for x, ref in (("BRT_tgb", "BRT_naive"), ("NN_occ_effort", "NN_an_ssdl"), ("BRT_tgb", "nearest_record")):
        d = g[x] - g[ref]
        out.append({"variant": var, "contrast": f"{x} - {ref}", "n_species": len(g), "delta": d.mean(),
                    "pct_better": (d > 0).mean(), "wilcoxon_p": wilcoxon(d).pvalue})
O = pd.DataFrame(out)
O.to_csv(RESULTS / "E17b_taiwan_occupancy_rescored.csv", index=False)
pd.DataFrame([gate]).to_csv(RESULTS / "E17b_taiwan_occupancy_gate.csv", index=False)
print(O.round(4).to_string(index=False))
