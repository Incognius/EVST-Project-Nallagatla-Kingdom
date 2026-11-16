import numpy as np
import pandas as pd
import statsmodels.api as sm

from evst.bias.discovery import AGG, forward_targets, visit_table
from evst.data import records
from evst.data.rasters import load_stack
from evst.paths import RESULTS

layers, tr, mask = load_stack()
H, W = mask.shape
Hn, Wn = -(-H // AGG), W // AGG + 1


def agg(x):
    p = np.full((Hn * AGG, Wn * AGG), np.nan)
    p[:H, :W] = np.where(mask, x, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(p.reshape(Hn, AGG, Wn, AGG), axis=(1, 3)).ravel()


ACC = ["d_road_all", "d_road_major", "d_town", "d_city", "d_reserve", "pop_log", "wc_built", "wc_tree", "elev"]
X = np.column_stack([agg(np.log1p(layers[k]) if k.startswith("d_") else layers[k]) for k in ACC])
cells = np.flatnonzero(np.isfinite(X).all(1))
Xz = (X[cells] - X[cells].mean(0)) / X[cells].std(0)
d = visit_table(records.load(), W)
Y0 = int(d.year.min())
rows = []
for pre, post in (((Y0, 2014), (2015, 2018)), ((Y0, 2018), (2019, 2026))):
    t = forward_targets(d, pre, post).reindex(cells, fill_value=0.0)
    sampled = t.U.values > 0
    base = np.column_stack([np.log1p(t.U), Xz])
    rich = np.log1p(t.S.values) - np.log1p(t.U.values) * 0
    resid_rich = rich - np.poly1d(np.polyfit(np.log1p(t.U.values[sampled]), rich[sampled], 2))(np.log1p(t.U.values))
    for name, extra in (("log_observed_richness", rich), ("richness_beyond_effort", resid_rich)):
        Xm = sm.add_constant(np.column_stack([base, extra])[sampled])
        fit = sm.GLM(t.m.values[sampled], Xm, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        lo, hi = fit.conf_int()[-1]
        rows.append({"pre": f"{pre[0]}-{pre[1]}", "post": f"{post[0]}-{post[1]}", "predictor": name,
                     "coef": fit.params[-1], "ci_lo": lo, "ci_hi": hi, "p": fit.pvalues[-1],
                     "rate_ratio_per_sd": float(np.exp(fit.params[-1] * extra[sampled].std())), "n_cells": int(sampled.sum())})
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "E08_observers_follow_richness.csv", index=False)
print(res.round(4).to_string(index=False))
