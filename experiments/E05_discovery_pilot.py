import argparse
import json

import numpy as np
import pandas as pd
import rasterio
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from evst.bias.discovery import AGG, DiscoveryModel, fit_phi, forward_targets, nb_logpmf, shen_chao, visit_table
from evst.data import records
from evst.data.rasters import load_stack
from evst.paths import PROCESSED, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--group", default="all")
ap.add_argument("--rff", type=int, default=24)
ap.add_argument("--prior-sd", type=float, default=1.0)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--tag", default="pilot")
a = ap.parse_args()

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
land = np.isfinite(E).all(1) & np.isfinite(Acc).all(1)
cells = np.flatnonzero(land)
Ez = PCA(8, random_state=0).fit_transform((E[cells] - E[cells].mean(0)) / E[cells].std(0))
Az = (Acc[cells] - Acc[cells].mean(0)) / Acc[cells].std(0)
r, c = np.divmod(cells, Wn)
rng = np.random.default_rng(0)
Om = rng.normal(0, 1 / 12.0, (2, a.rff))
ph = rng.uniform(0, 2 * np.pi, a.rff)
RFF = np.sqrt(2 / a.rff) * np.cos(np.column_stack([r, c]) @ Om + ph)
sdm = {}
for mname in ("GLM", "BRT"):
    for corr in ("naive", "tgb"):
        with rasterio.open(PROCESSED / f"E04_richness_{mname}_{corr}.tif") as s:
            v = s.read(1).ravel()[cells]
        v = np.nan_to_num(v, nan=np.nanmean(v))
        sdm[f"{mname}_{corr}"] = (v - v.mean()) / v.std()

df = records.load()
if a.group != "all":
    df = df[df.group == a.group]
d = visit_table(df, W)
pos = pd.Series(np.arange(len(cells)), index=cells)


def table(pre, post):
    t = forward_targets(d, pre, post).reindex(cells, fill_value=0.0)
    return t


def inc_arrays(pre):
    p = d[(d.year >= pre[0]) & (d.year <= pre[1])]
    inc = p.groupby(["c5", "species"]).visit.nunique().rename("Y").reset_index()
    inc = inc[inc.c5.isin(pos.index)]
    return pos[inc.c5].to_numpy(), inc.Y.to_numpy(float)


Y0 = int(d.year.min())
CAL = ((Y0, 2014), (2015, 2018))
FWD = ((Y0, 2018), (2019, 2026))


def design(extra=None):
    Z = np.column_stack([np.ones(len(cells)), Ez, RFF] + ([extra[:, None]] if extra is not None else []))
    Zd = np.column_stack([np.ones(len(cells)), Az])
    return Z, Zd


def run_discovery(split, extra=None):
    t = table(*split)
    ci, Y = inc_arrays(split[0])
    Z, Zd = design(extra)
    m = DiscoveryModel(prior_sd=a.prior_sd).fit(Z, Zd, t.U.values, t.S.values, ci, Y)
    return t, m.expected_new(Z, Zd, t.U.values, t.m.values)


def glm_feats(t, covs=False):
    X = np.column_stack([np.log1p(t.m), np.log1p(t.U), np.log1p(t.S), np.log1p(t.Q1)])
    if covs:
        X = np.column_stack([X, Ez, Az, RFF])
    return sm.add_constant(X, has_constant="add")


tc, tf = table(*CAL), table(*FWD)
evc, evf = (tc.m > 0).values, (tf.m > 0).values
preds, phis = {}, {}

for name, covs in (("effort_glm", False), ("effort_glm_covs", True)):
    Xc, Xf = glm_feats(tc, covs), glm_feats(tf, covs)
    fit = sm.GLM(tc.new.values[evc], Xc[evc], family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    phis[name] = fit_phi(tc.new.values[evc], fit.predict(Xc[evc]))
    preds[name] = fit.predict(Xf)

sc_c = shen_chao(tc.U, tc.S, tc.Q1, tc.Q2, tc.m)
phis["shen_chao"] = fit_phi(tc.new.values[evc], sc_c[evc] + 0.05)
preds["shen_chao"] = shen_chao(tf.U, tf.S, tf.Q1, tf.Q2, tf.m) + 0.05

_, mu_c = run_discovery(CAL)
phi_b = fit_phi(tc.new.values[evc], mu_c[evc])
_, preds["bayes_discovery"] = run_discovery(FWD)
phis["bayes_discovery"] = phi_b
for k, v in sdm.items():
    _, preds[f"bayes_discovery+sdm_{k}"] = run_discovery(FWD, v)
    phis[f"bayes_discovery+sdm_{k}"] = phi_b

y = tf.new.values[evf]
U = tf.U.values[evf]
strata = {"all": np.ones(evf.sum(), bool), "unsampled_pre (U=0)": U == 0, "sparse_pre (1-5 visits)": (U >= 1) & (U <= 5),
          "well_sampled_pre (>5 visits)": U > 5}
ll = {k: nb_logpmf(y, v[evf], phis[k]) for k, v in preds.items()}
base = "effort_glm_covs"
rows = []
for k in preds:
    for sname, sm_ in strata.items():
        if sm_.sum() < 10:
            continue
        diff = ll[k][sm_] - ll[base][sm_]
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(a.boot)]
        rows.append({"model": k, "stratum": sname, "n_cells": int(sm_.sum()), "mean_logscore": ll[k][sm_].mean(),
                     "delta_vs_" + base: diff.mean(), "ci_lo": np.quantile(bs, 0.025), "ci_hi": np.quantile(bs, 0.975),
                     "spearman": spearmanr(preds[k][evf][sm_], y[sm_])[0], "phi": phis[k]})
res = pd.DataFrame(rows)
res.to_csv(RESULTS / f"E05_discovery_{a.tag}_{a.group}.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
json.dump({"cells_eval": int(evf.sum()), "pre_years": FWD[0], "post_years": FWD[1], "group": a.group},
          open(RESULTS / f"E05_discovery_{a.tag}_{a.group}_meta.json", "w"))
