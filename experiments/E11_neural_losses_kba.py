import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from evst.data import kba
from evst.models import multisdm
from evst.paths import PROCESSED, RESULTS
from evst.validate.metrics import auc

ap = argparse.ArgumentParser()
ap.add_argument("--losses", default=",".join(multisdm.LOSSES))
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--steps", type=int, default=1500)
a = ap.parse_args()

S = kba.build()
X = S["X"]
Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)
dev = "cuda"
Xland = torch.as_tensor(Xz, device=dev)
b = S["birds"]
tg = np.sort(b.li.unique())
tpos = pd.Series(np.arange(len(tg)), index=tg)
sidx = pd.Series(np.arange(len(S["cand"])), index=S["cand"])
bc = b[b.species.isin(S["cand"])]
Ytg = np.zeros((len(tg), len(S["cand"])), np.float32)
Ytg[tpos[bc.li].to_numpy(), sidx[bc.species].to_numpy()] = 1
Utg = b.groupby("li").visit.nunique().reindex(tg).to_numpy(np.float32)
Yt, Ut = torch.as_tensor(Ytg, device=dev), torch.as_tensor(Utg, device=dev)
tg_t = torch.as_tensor(tg, device=dev)

Y, R = S["Y"], S["R"]
miss = R == 0
cell_scores = {}
rows = []
for loss in a.losses.split(","):
    t0 = time.time()
    acc = 0
    for seed in range(a.seeds):
        net = multisdm.fit(loss, Xland, tg_t, Yt, Ut, steps=a.steps, seed=seed)
        p = multisdm.score(net, loss, Xland)
        acc = acc + p / p.mean(0, keepdims=True)
    cs = kba.cell_mean(acc / a.seeds, S)
    cell_scores[loss] = cs
    np.save(PROCESSED / f"E11_cell_scores_{loss}.npy", cs)
    for j, sp in enumerate(S["cand"]):
        ym, sm = Y[miss[:, j], j], cs[miss[:, j], j]
        rows.append({"species": sp, "loss": loss, "endemic": bool(S["endemic"][j]),
                     "auc_missing": auc(ym, sm) if 5 <= ym.sum() <= len(ym) - 5 else np.nan,
                     "auc_all": auc(Y[:, j], cs[:, j]) if 5 <= Y[:, j].sum() <= len(Y) - 5 else np.nan})
    print(loss, f"{time.time() - t0:.0f}s", flush=True)

D = pd.DataFrame(rows)
D.to_csv(RESULTS / "E11_neural_species_auc.csv", index=False)
B = pd.read_csv(RESULTS / "E09b_kba_unrecorded_species_auc.csv").set_index("species")
W = D.pivot_table(index="species", columns="loss", values="auc_missing")
W = W.join(B[["BRT_tgb", "BRT_naive", "GLM_tgb", "nearest_record"]], how="inner").dropna()
print(f"species with AUC on missing pairs: {len(W)}")
print(W.mean().round(4).sort_values(ascending=False).to_string())
summ = []
for col in W.columns:
    for ref in ("BRT_tgb", "an_ssdl", "maxent_random"):
        if col == ref or ref not in W:
            continue
        d = W[col] - W[ref]
        summ.append({"model": col, "ref": ref, "mean_auc": W[col].mean(), "delta": d.mean(), "pct_better": (d > 0).mean(),
                     "wilcoxon_p": wilcoxon(d).pvalue})
Sm = pd.DataFrame(summ)
Sm.to_csv(RESULTS / "E11_neural_summary.csv", index=False)
print(Sm.round(4).to_string(index=False))
end = W.index.isin([s for s, e in zip(S["cand"], S["endemic"]) if e])
print("endemics:", W[end].mean().round(3).to_dict())
