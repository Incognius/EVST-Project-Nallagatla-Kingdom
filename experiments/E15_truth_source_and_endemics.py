import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

from evst.data import kba
from evst.models import multisdm
from evst.models.sdm import BRT
from evst.paths import RESULTS
from evst.validate.metrics import auc

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--train-end", type=int, default=2014)
ap.add_argument("--min-cells", type=int, default=10)
ap.add_argument("--jobs", type=int, default=10)
a = ap.parse_args()

S = kba.build(end=a.train_end, min_cells=a.min_cells)
L = kba.build(end=2026, start=a.train_end + 1, min_cells=1)
cand = S["cand"]
lpos = pd.Series(np.arange(len(L["cand"])), index=L["cand"])
later = np.zeros_like(S["Y"])
for j, sp in enumerate(cand):
    if sp in lpos.index:
        later[:, j] = L["R"][:, lpos[sp]]
Ul = L["U"]
Y, R = S["Y"], S["R"]
X = S["X"]
b = S["birds"]
tg = np.unique(b.li.to_numpy())
rng = np.random.default_rng(0)
bg = rng.choice(len(X), 20000, replace=False)
cellsets = b.groupby("species").li.unique()
pix_li, pix_k = S["pix_li"], S["pix_k"]


def brt(sp):
    pc = np.unique(cellsets[sp])
    out = {}
    for corr, bgc in (("naive", bg), ("tgb", tg)):
        bgc = np.setdiff1d(bgc, pc)
        m = BRT().fit(np.r_[X[pc], X[bgc]], np.r_[np.ones(len(pc)), np.zeros(len(bgc))])
        out[f"BRT_{corr}"] = m.predict(X[pix_li])
    return out


res = Parallel(n_jobs=a.jobs)(delayed(brt)(sp) for sp in cand)
scores = {}
for k in ("BRT_naive", "BRT_tgb"):
    pl = np.zeros((len(X), len(cand)), np.float32)
    pl[pix_li] = np.column_stack([r[k] for r in res])
    scores[k] = kba.cell_mean(pl, S)
Xz, tgn, Ytg, Utg = kba.training_arrays(S)
dev = "cuda"
T = [torch.as_tensor(v, device=dev) for v in (Xz, tgn, Ytg, Utg)]
for loss in ("an_ssdl", "occ_effort"):
    acc = 0
    for seed in range(5):
        net = multisdm.fit(loss, *T, seed=seed)
        p = multisdm.score(net, loss, T[0])
        acc = acc + p / p.mean(0, keepdims=True)
    scores[f"NN_{loss}"] = kba.cell_mean(acc / 5, S)

rows = []
for j, sp in enumerate(cand):
    unrec = R[:, j] == 0
    eff = unrec & (Ul > 0)
    for truth, m, y in (("kba_all_unrecorded", unrec, Y[unrec, j]), ("kba_same_cells", eff, Y[eff, j]),
                        ("inat_later_same_cells", eff, later[eff, j])):
        if 5 <= y.sum() <= len(y) - 5:
            rows.append({"species": sp, "truth": truth, "n": int(m.sum()), "prev": y.mean(),
                         **{k: auc(y, v[m, j]) for k, v in scores.items()}})
D = pd.DataFrame(rows)
D.to_csv(RESULTS / f"E15_truth_source_train{a.train_end}.csv", index=False)
summ = []
for truth, g in D.groupby("truth"):
    for x, ref in (("BRT_tgb", "BRT_naive"), ("NN_occ_effort", "NN_an_ssdl")):
        d = g[x] - g[ref]
        summ.append({"truth": truth, "contrast": f"{x} - {ref}", "n_species": len(g), "mean_ref": g[ref].mean(),
                     "mean_corrected": g[x].mean(), "delta": d.mean(), "pct_corrected_better": (d > 0).mean(),
                     "wilcoxon_p": wilcoxon(d).pvalue})
Sm = pd.DataFrame(summ)
Sm.to_csv(RESULTS / f"E15_truth_source_summary_train{a.train_end}.csv", index=False)
print(f"train <= {a.train_end}: {len(cand)} species, iNat records {len(b)}, KBA cells with later iNat effort {(Ul > 0).sum()}")
print(Sm.round(4).to_string(index=False))

F = kba.build()
E = pd.read_csv(RESULTS / "E09b_kba_unrecorded_species_auc.csv")
fpos = pd.Series(np.arange(len(F["cand"])), index=F["cand"])
E = E[E.species.isin(fpos.index)].copy()
jj = fpos[E.species].to_numpy()
Yf, Uf = F["Y"], F["U"]
E["gain"] = E.BRT_tgb - E.BRT_naive
E["endemic"] = F["endemic"][jj].astype(int)
E["exposure"] = [(Yf[:, j] * (Uf == 0)).sum() / max(Yf[:, j].sum(), 1) for j in jj]
E["log_train_cells"] = np.log(F["birds"].groupby("species").li.nunique().reindex(E.species).to_numpy())
E["kba_prev"] = Yf[:, jj].mean(0)
ce = kba.cell_mean(F["X"][:, [19]], F)[:, 0]
E["elev_mean"] = [(ce * Yf[:, j]).sum() / max(Yf[:, j].sum(), 1) / 1000 for j in jj]
fits = {}
for f in ("gain ~ endemic", "gain ~ endemic + exposure", "gain ~ endemic + exposure + log_train_cells + kba_prev",
          "gain ~ exposure + log_train_cells + kba_prev + elev_mean"):
    fits[f] = smf.ols(f, E).fit(cov_type="HC3")
    print("\n", f, "\n", fits[f].summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]].round(4).to_string())
perm = []
base = fits["gain ~ endemic + exposure + log_train_cells + kba_prev"].params["endemic"]
for i in range(2000):
    Ep = E.assign(endemic=np.random.default_rng(i).permutation(E.endemic.values))
    perm.append(smf.ols("gain ~ endemic + exposure + log_train_cells + kba_prev", Ep).fit().params["endemic"])
print("endemic coef adjusted:", round(base, 4), "permutation p:", round((np.abs(perm) >= abs(base)).mean(), 4))
print("exposure endemic vs not:", E.groupby("endemic").exposure.mean().round(3).to_dict())
E.to_csv(RESULTS / "E15_endemic_gain_covariates.csv", index=False)
