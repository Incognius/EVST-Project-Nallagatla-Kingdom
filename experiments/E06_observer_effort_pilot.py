import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter

from evst.bias.discovery import AGG, visit_table
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
land = np.isfinite(X).all(1)
cells = np.flatnonzero(land)
Xz = (X[cells] - X[cells].mean(0)) / X[cells].std(0)
Xd = sm.add_constant(Xz)

d = visit_table(records.load(), W)
vis = d.drop_duplicates(["visit", "c5"])[["visit", "c5", "recordedby", "year"]]
vis = vis[vis.c5.isin(cells)]
Y0 = int(vis.year.min())


def counts(lo, hi):
    v = vis[(vis.year >= lo) & (vis.year <= hi)]
    return v.c5.value_counts().reindex(cells, fill_value=0).to_numpy(float)


def grid(vec):
    g = np.zeros(Hn * Wn)
    g[cells] = vec
    return g.reshape(Hn, Wn)


land_g = grid(np.ones(len(cells)))


def smooth(vec, sigma):
    if sigma == 0:
        return vec
    num = gaussian_filter(grid(vec), sigma)
    den = gaussian_filter(land_g, sigma)
    return (num / np.maximum(den, 1e-9)).ravel()[cells]


def observer_kernel(lo, hi, sigma, recency):
    v = vis[(vis.year >= lo) & (vis.year <= hi)]
    tot = np.zeros(len(cells))
    pos = pd.Series(np.arange(len(cells)), index=cells)
    for o, g in v.groupby("recordedby"):
        cnt = np.bincount(pos[g.c5].to_numpy(), minlength=len(cells)).astype(float)
        share = cnt / cnt.sum()
        wt = cnt.sum() ** 0.5 * np.exp(-recency * (hi - g.year.max()))
        tot += wt * share
    return smooth(tot, sigma)


def norm(p):
    p = np.maximum(p, 1e-12)
    return p / p.sum()


def score(p, y):
    return float((y * np.log(norm(p))).sum() / y.sum())


def access_glm(y_pre):
    fit = sm.GLM(y_pre, Xd, family=sm.families.Poisson()).fit()
    return fit.predict(Xd)


def models(pre, params):
    y_pre = counts(*pre)
    acc = access_glm(y_pre)
    persist = smooth(y_pre, params["sp"]) + params["alpha"]
    ok = observer_kernel(*pre, params["so"], params["rec"])
    out = {"uniform": np.ones(len(cells)), "access_glm": acc, "persistence": persist,
           "persistence+access": norm(persist) ** (1 - params["wa"]) * norm(acc) ** params["wa"],
           "observer_kernel": ok,
           "observer+new_observers": (1 - params["wn"]) * norm(ok) + params["wn"] * norm(acc)}
    return out


cal_pre, cal_post = (Y0, 2014), (2015, 2018)
fwd_pre, fwd_post = (Y0, 2018), (2019, 2026)
y_cal = counts(*cal_post)
best, best_s = None, -np.inf
grid_search = []
for sp in (0, 1, 2):
    for so in (0, 1, 2, 4):
        for rec in (0.0, 0.3, 1.0):
            for wn in (0.1, 0.3, 0.5):
                prm = {"sp": sp, "alpha": 0.05, "so": so, "rec": rec, "wn": wn, "wa": 0.3}
                s = score(models(cal_pre, prm)["observer+new_observers"], y_cal)
                grid_search.append({**prm, "cal_score": s})
                if s > best_s:
                    best, best_s = prm, s
for wa in (0.1, 0.3, 0.5, 0.7):
    for alpha in (0.01, 0.05, 0.2, 1.0):
        prm = {**best, "wa": wa, "alpha": alpha}
        s = score(models(cal_pre, prm)["persistence+access"], y_cal)
        if s > score(models(cal_pre, best)["persistence+access"], y_cal):
            best = {**best, "wa": wa, "alpha": alpha}
y_fwd = counts(*fwd_post)
fm = models(fwd_pre, best)
rng = np.random.default_rng(0)
rows = []
ref = "persistence+access"
for k, p in fm.items():
    lp = np.log(norm(p))
    lr = np.log(norm(fm[ref]))
    per_cell = y_fwd * (lp - lr)
    bs = [per_cell[rng.integers(0, len(cells), len(cells))].sum() / y_fwd.sum() for _ in range(1000)]
    rows.append({"model": k, "logscore_per_visit": score(p, y_fwd), "delta_vs_" + ref: per_cell.sum() / y_fwd.sum(),
                 "ci_lo": np.quantile(bs, 0.025), "ci_hi": np.quantile(bs, 0.975),
                 "top10pct_capture": float(y_fwd[np.argsort(-p)[: len(cells) // 10]].sum() / y_fwd.sum())})
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "E06_observer_effort_pilot.csv", index=False)
json.dump(best, open(RESULTS / "E06_observer_effort_params.json", "w"))
pd.DataFrame(grid_search).to_csv(RESULTS / "E06_observer_effort_grid.csv", index=False)
print("chosen on calibration split:", best)
print(res.round(4).to_string(index=False))
