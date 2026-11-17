import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

from evst.data import kba
from evst.models import multisdm
from evst.paths import RESULTS
from evst.validate.metrics import auc

S = kba.build()
Xz, tg, Ytg, Utg = kba.training_arrays(S)
dev = "cuda"
eff = np.zeros(len(Xz), np.float32)
eff[tg] = np.log1p(Utg)
A = np.column_stack([S["A"], eff])
Az = (A - A.mean(0)) / (A.std(0) + 1e-6)
names = ["d_road_all", "d_road_major", "d_town", "d_reserve", "pop_log", "inat_effort"]
Xl, Al = torch.as_tensor(Xz, device=dev), torch.as_tensor(Az.astype(np.float32), device=dev)
tg_t, Yt, Ut = torch.as_tensor(tg, device=dev), torch.as_tensor(Ytg, device=dev), torch.as_tensor(Utg, device=dev)
Y, R = S["Y"], S["R"]
rng = np.random.default_rng(0)
probe = rng.choice(len(Xz), 20000, replace=False)


def leak(Z):
    return [cross_val_score(RidgeCV(alphas=np.logspace(-2, 3, 8)), Z[probe], Az[probe, k], cv=5, scoring="r2").mean()
            for k in range(Az.shape[1])]


leak_rows = [{"variant": "raw_env_inputs", **dict(zip(names, leak(Xz)))}]
sp_rows = []
for loss in ("an_ssdl", "occ_effort"):
    for lam in (0.0, 0.1, 0.3, 1.0):
        acc, lk = 0, []
        for seed in range(5):
            net = multisdm.fit_adv(loss, Xl, Al, tg_t, Yt, Ut, adv=lam, seed=seed)
            p = multisdm.score(net, loss, Xl)
            acc = acc + p / p.mean(0, keepdims=True)
            if seed < 2:
                lk.append(leak(multisdm.embed(net, Xl)))
        leak_rows.append({"variant": f"{loss}_adv{lam}", **dict(zip(names, np.mean(lk, 0)))})
        cs = kba.cell_mean(acc / 5, S)
        for j, sp in enumerate(S["cand"]):
            m = R[:, j] == 0
            y = Y[m, j]
            if 5 <= y.sum() <= len(y) - 5:
                sp_rows.append({"species": sp, "loss": loss, "adv": lam, "auc_missing": auc(y, cs[m, j]),
                                "auc_all": auc(Y[:, j], cs[:, j])})
        print(loss, lam, leak_rows[-1], flush=True)

L = pd.DataFrame(leak_rows)
D = pd.DataFrame(sp_rows)
L.to_csv(RESULTS / "E14_embedding_leakage.csv", index=False)
D.to_csv(RESULTS / "E14_adversarial_species_auc.csv", index=False)
print(L.round(3).to_string(index=False))
out = []
for loss, g in D.groupby("loss"):
    w = g.pivot_table(index="species", columns="adv", values="auc_missing").dropna()
    for lam in w.columns:
        d = w[lam] - w[0.0]
        out.append({"loss": loss, "adv": lam, "mean_auc_missing": w[lam].mean(), "delta_vs_no_adv": d.mean(),
                    "pct_better": (d > 0).mean(), "wilcoxon_p": wilcoxon(d).pvalue if lam > 0 else np.nan})
O = pd.DataFrame(out)
O.to_csv(RESULTS / "E14_adversarial_summary.csv", index=False)
print(O.round(4).to_string(index=False))
