import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

from evst.data import kba
from evst.models import multisdm
from evst.paths import RESULTS

S = kba.build()
Xz, tg, Ytg, Utg = kba.training_arrays(S)
eff = np.zeros(len(Xz), np.float32)
eff[tg] = np.log1p(Utg)
A = np.column_stack([S["A"], eff])
Az = (A - A.mean(0)) / (A.std(0) + 1e-6)
names = ["d_road_all", "d_road_major", "d_town", "d_reserve", "pop_log", "inat_effort"]
probe = np.random.default_rng(0).choice(len(Xz), 20000, replace=False)
Xl = torch.as_tensor(Xz, device="cuda")
rows = []
for seed in range(2):
    torch.manual_seed(seed)
    Z = multisdm.embed(multisdm.Net(Xz.shape[1], len(S["cand"])).cuda(), Xl)
    rows.append({"variant": f"untrained_net_seed{seed}", **{n: cross_val_score(RidgeCV(alphas=np.logspace(-2, 3, 8)), Z[probe], Az[probe, k], cv=5, scoring="r2").mean() for k, n in enumerate(names)}})
rows.append({"variant": "gbm_on_raw_env", **{n: cross_val_score(HistGradientBoostingRegressor(max_iter=200), Xz[probe], Az[probe, k], cv=5, scoring="r2").mean() for k, n in enumerate(names)}})
D = pd.DataFrame(rows)
D.to_csv(RESULTS / "E14b_leakage_controls.csv", index=False)
print(D.round(3).to_string(index=False))
