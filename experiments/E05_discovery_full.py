import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logsumexp
from scipy.stats import nbinom, spearmanr
from sklearn.decomposition import PCA

from evst.bias.discovery import AGG, DiscoveryModel, fit_phi, forward_targets, nb_logpmf, shen_chao, visit_table
from evst.data import records
from evst.data.rasters import load_stack
from evst.paths import FIGURES, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--groups", default="all,birds,butterflies_moths,odonates,plants,reptiles,amphibians,mammals")
ap.add_argument("--origins", default="2016,2018,2020,2022")
ap.add_argument("--window", type=int, default=3)
ap.add_argument("--boot", type=int, default=1000)
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
cells = np.flatnonzero(np.isfinite(E).all(1) & np.isfinite(Acc).all(1))
Ez = PCA(8, random_state=0).fit_transform((E[cells] - E[cells].mean(0)) / E[cells].std(0))
Az = (Acc[cells] - Acc[cells].mean(0)) / Acc[cells].std(0)
r, c = np.divmod(cells, Wn)
rng = np.random.default_rng(0)
Om, ph = rng.normal(0, 1 / 12.0, (2, 24)), rng.uniform(0, 2 * np.pi, 24)
RFF = np.sqrt(2 / 24) * np.cos(np.column_stack([r, c]) @ Om + ph)
ONE = np.ones((len(cells), 1))
VARIANTS = {
    "bayes": (np.hstack([ONE, Ez, RFF]), np.hstack([ONE, Az])),
    "bayes_no_spatial": (np.hstack([ONE, Ez]), np.hstack([ONE, Az])),
    "bayes_no_habitat": (np.hstack([ONE, RFF]), np.hstack([ONE, Az])),
    "bayes_no_detection_covs": (np.hstack([ONE, Ez, RFF]), ONE),
    "bayes_intercepts_only": (ONE, ONE),
}
all_rec = records.load()
pos = pd.Series(np.arange(len(cells)), index=cells)


def inc_arrays(d, pre):
    p = d[(d.year >= pre[0]) & (d.year <= pre[1])]
    inc = p.groupby(["c5", "species"]).visit.nunique().rename("Y").reset_index()
    inc = inc[inc.c5.isin(pos.index)]
    return pos[inc.c5].to_numpy(), inc.Y.to_numpy(float)


def nb_q(mu, phi, q):
    return nbinom.ppf(q, phi, phi / (phi + np.maximum(mu, 1e-9)))


def glm_X(t, covs):
    X = np.column_stack([np.log1p(t.m), np.log1p(t.U), np.log1p(t.S), np.log1p(t.Q1)])
    if covs:
        X = np.column_stack([X, Ez, Az, RFF])
    return sm.add_constant(X, has_constant="add")


score_rows, cov_rows, val_rows = [], [], []
for group in a.groups.split(","):
    df = all_rec if group == "all" else all_rec[all_rec.group == group]
    d = visit_table(df, W)
    Y0 = int(d.year.min())
    origins = [int(o) for o in a.origins.split(",")] if group == "all" else [2018]
    for T in origins:
        w = a.window
        CAL, FWD = ((Y0, T - w), (T - w + 1, T)), ((Y0, T), (T + 1, T + w))
        tc = forward_targets(d, *CAL).reindex(cells, fill_value=0.0)
        tf = forward_targets(d, *FWD).reindex(cells, fill_value=0.0)
        evc, evf = (tc.m > 0).values, (tf.m > 0).values
        if evf.sum() < 50 or tc.new.values[evc].sum() < 20:
            continue
        mu, phis, mix = {}, {}, {}
        for name, covs in (("effort_glm", False), ("effort_glm_covs", True)):
            fit = sm.GLM(tc.new.values[evc], glm_X(tc, covs)[evc], family=sm.families.NegativeBinomial(alpha=1.0)).fit()
            phis[name] = fit_phi(tc.new.values[evc], fit.predict(glm_X(tc, covs)[evc]))
            mu[name] = fit.predict(glm_X(tf, covs))
        phis["shen_chao"] = fit_phi(tc.new.values[evc], shen_chao(tc.U, tc.S, tc.Q1, tc.Q2, tc.m)[evc] + 0.05)
        mu["shen_chao"] = shen_chao(tf.U, tf.S, tf.Q1, tf.Q2, tf.m) + 0.05
        for vname, (Z, Zd) in VARIANTS.items():
            ci_c, Yc = inc_arrays(d, CAL[0])
            mc = DiscoveryModel().fit(Z, Zd, tc.U.values, tc.S.values, ci_c, Yc)
            phis[vname] = fit_phi(tc.new.values[evc], mc.expected_new(Z, Zd, tc.U.values, tc.m.values)[evc])
            ci_f, Yf = inc_arrays(d, FWD[0])
            mf = DiscoveryModel().fit(Z, Zd, tf.U.values, tf.S.values, ci_f, Yf)
            mu[vname] = mf.expected_new(Z, Zd, tf.U.values, tf.m.values)
            if vname == "bayes":
                mf.laplace()
                mix["bayes_laplace"] = mf.posterior_new(Z, Zd, tf.U.values, tf.m.values, draws=200)
        y = tf.new.values[evf]
        U = tf.U.values[evf]
        ll = {k: nb_logpmf(y, v[evf], phis[k]) for k, v in mu.items()}
        draws = mix["bayes_laplace"][:, evf]
        ll["bayes_laplace"] = logsumexp(np.stack([nb_logpmf(y, dr, phis["bayes"]) for dr in draws]), 0) - np.log(len(draws))
        strata = {"all": np.ones(len(y), bool), "unsampled_pre": U == 0, "sparse_pre_1to5": (U >= 1) & (U <= 5),
                  "well_sampled_pre_gt5": U > 5}
        for k in ll:
            for sname, sm_ in strata.items():
                if sm_.sum() < 20:
                    continue
                diff = ll[k][sm_] - ll["effort_glm"][sm_]
                bs = [rng.choice(diff, len(diff)).mean() for _ in range(a.boot)]
                score_rows.append({"group": group, "origin": T, "model": k, "stratum": sname, "n_cells": int(sm_.sum()),
                                   "logscore": ll[k][sm_].mean(), "delta_vs_effort_glm": diff.mean(),
                                   "ci_lo": np.quantile(bs, 0.025), "ci_hi": np.quantile(bs, 0.975)})
        for k in ("effort_glm", "effort_glm_covs", "bayes"):
            lo, hi = nb_q(mu[k][evf], phis[k], 0.1), nb_q(mu[k][evf], phis[k], 0.9)
            cov_rows.append({"group": group, "origin": T, "model": k, "coverage_80": float(((y >= lo) & (y <= hi)).mean()),
                             "pit_lt_0.1": float((y < lo).mean()), "pit_gt_0.9": float((y > hi).mean())})
        sim = np.stack([nbinom.rvs(phis["bayes"], phis["bayes"] / (phis["bayes"] + np.maximum(dr, 1e-9)), random_state=i)
                        for i, dr in enumerate(draws)])
        lo, hi = np.quantile(sim, 0.1, axis=0), np.quantile(sim, 0.9, axis=0)
        cov_rows.append({"group": group, "origin": T, "model": "bayes_laplace", "coverage_80": float(((y >= lo) & (y <= hi)).mean()),
                         "pit_lt_0.1": float((y < lo).mean()), "pit_gt_0.9": float((y > hi).mean())})
        m = tf.m.values[evf]
        sel = m >= 5
        if sel.sum() >= 30:
            realized = y[sel] / m[sel]
            for k in ("effort_glm", "effort_glm_covs", "shen_chao", "bayes"):
                rate = mu[k][evf][sel] / m[sel]
                top = np.argsort(-rate)[: max(1, sel.sum() // 10)]
                val_rows.append({"group": group, "origin": T, "model": k, "n_cells": int(sel.sum()),
                                 "spearman_rate": spearmanr(rate, realized)[0],
                                 "top10_realized_rate": realized[top].mean(), "mean_realized_rate": realized.mean()})
        print(group, T, {k: round(float(v.mean()), 3) for k, v in ll.items()}, flush=True)

S = pd.DataFrame(score_rows)
C = pd.DataFrame(cov_rows)
V = pd.DataFrame(val_rows)
S.to_csv(RESULTS / "E05_full_scores.csv", index=False)
C.to_csv(RESULTS / "E05_full_coverage.csv", index=False)
V.to_csv(RESULTS / "E05_full_discovery_value.csv", index=False)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
sa = S[(S.group == "all") & (S.stratum == "all")]
for k, col in (("bayes", "#D2601A"), ("bayes_laplace", "#8E3510"), ("effort_glm_covs", "#7E7C72"), ("bayes_intercepts_only", "#4F6B35")):
    q = sa[sa.model == k].sort_values("origin")
    ax[0].errorbar(q.origin, q.delta_vs_effort_glm, yerr=[q.delta_vs_effort_glm - q.ci_lo, q.ci_hi - q.delta_vs_effort_glm],
                   marker="o", capsize=3, color=col, label=k)
ax[0].axhline(0, c="k", lw=0.6)
ax[0].set_xlabel("forecast origin (train <= year, test next 3 years)")
ax[0].set_ylabel("log score gain vs effort GLM (nats/cell)")
ax[0].legend(fontsize=7)
sg = S[(S.origin == 2018) & (S.stratum == "all") & (S.model == "bayes")].set_index("group").delta_vs_effort_glm
sg.plot.barh(ax=ax[1], color="#D2601A")
ax[1].axvline(0, c="k", lw=0.6)
ax[1].set_xlabel("log score gain vs effort GLM, origin 2018")
fig.tight_layout()
fig.savefig(FIGURES / "E05_discovery_full.png")
print(S[(S.group == "all") & (S.stratum == "all")].pivot_table(index="model", columns="origin", values="delta_vs_effort_glm").round(3))
print(C.groupby("model").coverage_80.mean().round(3))
print(V.groupby("model")[["spearman_rate", "top10_realized_rate", "mean_realized_rate"]].mean().round(3))
