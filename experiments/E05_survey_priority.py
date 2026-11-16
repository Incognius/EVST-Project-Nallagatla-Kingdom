import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from evst.bias.discovery import AGG, DiscoveryModel, forward_targets, period_summary, shen_chao, visit_table
from evst.data import records
from evst.data.rasters import load_stack
from evst.paths import FIGURES, PROCESSED, RESULTS

layers, tr, mask = load_stack()
H, W = mask.shape
Hn, Wn = -(-H // AGG), W // AGG + 1


def agg(x):
    p = np.full((Hn * AGG, Wn * AGG), np.nan)
    p[:H, :W] = np.where(mask, x, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(p.reshape(Hn, AGG, Wn, AGG), axis=(1, 3)).ravel()


ENV = [f"bio{i:02d}" for i in range(1, 20)] + ["elev", "wc_tree", "wc_grass", "wc_crop", "wc_built", "wc_shrub", "wc_water"]
ACC = ["d_road_all", "d_road_major", "d_town", "d_reserve", "pop_log"]
E = np.column_stack([agg(layers[k]) for k in ENV])
Acc = np.column_stack([agg(np.log1p(layers[k]) if k.startswith("d_") else layers[k]) for k in ACC])
cells = np.flatnonzero(np.isfinite(E).all(1) & np.isfinite(Acc).all(1))
Ez = PCA(8, random_state=0).fit_transform((E[cells] - E[cells].mean(0)) / E[cells].std(0))
Az = (Acc[cells] - Acc[cells].mean(0)) / Acc[cells].std(0)
rr, cc = np.divmod(cells, Wn)
_r = np.random.default_rng(0)
Om, ph = _r.normal(0, 1 / 12.0, (2, 24)), _r.uniform(0, 2 * np.pi, 24)
RFF = np.sqrt(2 / 24) * np.cos(np.column_stack([rr, cc]) @ Om + ph)
pos = pd.Series(np.arange(len(cells)), index=cells)
d = visit_table(records.load(), W)
Y0 = int(d.year.min())
ONE = np.ones((len(cells), 1))
ZF, ZDF = np.hstack([ONE, Ez, RFF]), np.hstack([ONE, Az])
rng = np.random.default_rng(1)


def inc_arrays(pre):
    p = d[(d.year >= pre[0]) & (d.year <= pre[1])]
    inc = p.groupby(["c5", "species"]).visit.nunique().rename("Y").reset_index()
    inc = inc[inc.c5.isin(pos.index)]
    return pos[inc.c5].to_numpy(), inc.Y.to_numpy(float)


def X(t):
    return sm.add_constant(np.column_stack([np.log1p(t.m), np.log1p(t.U), np.log1p(t.S), np.log1p(t.Q1)]), has_constant="add")


rows = []
for T in (2016, 2018, 2020, 2022):
    CAL, FWD = ((Y0, T - 3), (T - 2, T)), ((Y0, T), (T + 1, T + 3))
    tc = forward_targets(d, *CAL).reindex(cells, fill_value=0.0)
    tf = forward_targets(d, *FWD).reindex(cells, fill_value=0.0)
    evc = (tc.m > 0).values
    glm = sm.GLM(tc.new.values[evc], X(tc)[evc], family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    ci, Y = inc_arrays(FWD[0])
    bm = DiscoveryModel().fit(ONE, ONE, tf.U.values, tf.S.values, ci, Y)
    bf = DiscoveryModel().fit(ZF, ZDF, tf.U.values, tf.S.values, ci, Y)
    m = tf.m.values
    sel = m >= 5
    realized = tf.new.values[sel] / m[sel]
    preds = {"bayes_full": bf.expected_new(ZF, ZDF, tf.U.values, m)[sel] / m[sel],
             "bayes_intercepts": bm.expected_new(ONE, ONE, tf.U.values, m)[sel] / m[sel],
             "effort_glm": glm.predict(X(tf))[sel] / m[sel],
             "shen_chao": shen_chao(tf.U, tf.S, tf.Q1, tf.Q2, tf.m)[sel] / m[sel],
             "fewest_prior_visits": -tf.U.values[sel]}
    n = sel.sum()
    k = max(1, n // 10)
    for name, p in preds.items():
        top = np.argsort(-p)[:k]
        bs_r, bs_top = [], []
        for _ in range(1000):
            i = rng.integers(0, n, n)
            bs_r.append(spearmanr(p[i], realized[i])[0])
            ti = i[np.argsort(-p[i])[:k]]
            bs_top.append(realized[ti].mean() / realized[i].mean())
        rows.append({"origin": T, "indicator": name, "n_cells": int(n), "spearman": spearmanr(p, realized)[0],
                     "sp_lo": np.nanquantile(bs_r, 0.025), "sp_hi": np.nanquantile(bs_r, 0.975),
                     "top10_lift": realized[top].mean() / realized.mean(),
                     "lift_lo": np.quantile(bs_top, 0.025), "lift_hi": np.quantile(bs_top, 0.975)})
R = pd.DataFrame(rows)
R.to_csv(RESULTS / "E05_survey_priority_validation.csv", index=False)
print(R.round(3).to_string(index=False))

Uall, incall = period_summary(d, Y0, 2026)
t = pd.DataFrame({"U": Uall, "S": incall.groupby("c5").size()}).reindex(cells, fill_value=0.0)
ci, Y = inc_arrays((Y0, 2026))
bm = DiscoveryModel().fit(ZF, ZDF, t.U.values, t.S.values, ci, Y).laplace()
gain10 = bm.posterior_new(ZF, ZDF, t.U.values, np.full(len(cells), 10.0), draws=400)
mean10, lo10, hi10 = gain10.mean(0), np.quantile(gain10, 0.05, 0), np.quantile(gain10, 0.95, 0)
img = np.full(Hn * Wn, np.nan)
img[cells] = mean10
prof = dict(driver="GTiff", height=Hn, width=Wn, count=3, dtype="float32", crs="EPSG:32643",
            transform=tr * tr.scale(AGG, AGG), nodata=np.nan, compress="deflate")
with rasterio.open(PROCESSED / "E05_expected_new_per_10_visits.tif", "w", **prof) as dst:
    for b, v in enumerate((mean10, lo10, hi10), 1):
        g = np.full(Hn * Wn, np.nan)
        g[cells] = v
        dst.write(g.reshape(Hn, Wn).astype("float32"), b)
pd.DataFrame({"cell": cells, "visits_to_2026": t.U.values, "species_to_2026": t.S.values, "expected_new_10": mean10,
              "q05": lo10, "q95": hi10}).to_csv(RESULTS / "E05_survey_priority_cells.csv", index=False)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
fig, ax = plt.subplots(1, 3, figsize=(9, 6))
vis = np.full(Hn * Wn, np.nan)
vis[cells] = np.log10(t.U.values + 1)
for a_, im, ttl, cm in ((ax[0], vis, "log10 visits to 2026", "magma"), (ax[1], img, "expected new species\nin next 10 visits", "viridis"),
                        (ax[2], np.where(np.isfinite(img), (img >= np.nanquantile(img, 0.9)).astype(float), np.nan),
                         "top 10% survey priority", "Oranges")):
    h = a_.imshow(im.reshape(Hn, Wn), cmap=cm)
    a_.set_title(ttl)
    a_.axis("off")
    plt.colorbar(h, ax=a_, shrink=0.5)
fig.tight_layout()
fig.savefig(FIGURES / "E05_survey_priority_map.png")
print("posterior band width (q95-q05) median:", float(np.median(hi10 - lo10)))
