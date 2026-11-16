import numpy as np
import pandas as pd
import statsmodels.api as sm
from pyproj import Transformer
from scipy.stats import spearmanr

from evst.bias.discovery import DiscoveryModel, fit_phi, nb_logpmf, p_undetected, shen_chao
from evst.paths import RAW, RESULTS

G = RAW / "glc24"
CELL = 10_000
tf = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
po = pd.read_csv(G / "GLC24_P0_metadata_train.csv", usecols=["publisher", "year", "lat", "lon", "date", "speciesId"])
x, y = tf.transform(po.lon.values, po.lat.values)
po["cx"], po["cy"] = (x // CELL).astype(int), (y // CELL).astype(int)
po["c"] = po.cx * 100000 + po.cy
po["visit"] = po.publisher + "|" + po.date.astype(str)
pa = pd.read_csv(G / "GLC24_PA_metadata_train.csv", usecols=["lat", "lon", "surveyId", "speciesId"])
x, y = tf.transform(pa.lon.values, pa.lat.values)
pa["c"] = (x // CELL).astype(int) * 100000 + (y // CELL).astype(int)
cells = np.array(sorted(po.c.unique()))
pos = pd.Series(np.arange(len(cells)), index=cells)
cxy = np.column_stack([cells // 100000, cells % 100000]).astype(float)
rng = np.random.default_rng(0)
Om, ph = rng.normal(0, 1 / 30.0, (2, 32)), rng.uniform(0, 2 * np.pi, 32)
Z = np.hstack([np.ones((len(cells), 1)), np.sqrt(2 / 32) * np.cos(cxy @ Om + ph)])
Zd = np.ones((len(cells), 1))


def summary(lo, hi):
    p = po[(po.year >= lo) & (po.year <= hi)]
    inc = p.groupby(["c", "speciesId"]).visit.nunique().rename("Y").reset_index()
    U = p.groupby("c").visit.nunique().reindex(cells, fill_value=0).astype(float)
    S = inc.groupby("c").size().reindex(cells, fill_value=0).astype(float)
    Q1 = inc[inc.Y == 1].groupby("c").size().reindex(cells, fill_value=0).astype(float)
    Q2 = inc[inc.Y == 2].groupby("c").size().reindex(cells, fill_value=0).astype(float)
    return U, S, Q1, Q2, inc


def target(pre, post):
    _, _, _, _, inc = summary(*pre)
    m, _, _, _, inc2 = summary(*post)
    seen = set(zip(inc.c, inc.speciesId))
    new = inc2[[(c, s) not in seen for c, s in zip(inc2.c, inc2.speciesId)]].groupby("c").size()
    return m, new.reindex(cells, fill_value=0).astype(float)


def bayes(pre):
    U, S, _, _, inc = summary(*pre)
    return DiscoveryModel().fit(Z, Zd, U.values, S.values, pos[inc.c].to_numpy(), inc.Y.to_numpy(float)), U


def glmX(U, S, Q1, m):
    return sm.add_constant(np.column_stack([np.log1p(m), np.log1p(U), np.log1p(S), np.log1p(Q1)]), has_constant="add")


rows = []
for T in (2018, 2019, 2020):
    cal_pre, cal_post, fwd_pre, fwd_post = (2017, T - 1), (T, T), (2017, T), (T + 1, T + 1)
    Uc, Sc, Q1c, Q2c, _ = summary(*cal_pre)
    mc, nc = target(cal_pre, cal_post)
    Uf, Sf, Q1f, Q2f, _ = summary(*fwd_pre)
    mf, nf = target(fwd_pre, fwd_post)
    evc, evf = (mc > 0).values, (mf > 0).values
    mu, phi = {}, {}
    g = sm.GLM(nc.values[evc], glmX(Uc, Sc, Q1c, mc)[evc], family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    phi["effort_glm"] = fit_phi(nc.values[evc], g.predict(glmX(Uc, Sc, Q1c, mc)[evc]))
    mu["effort_glm"] = g.predict(glmX(Uf, Sf, Q1f, mf))
    phi["shen_chao"] = fit_phi(nc.values[evc], shen_chao(Uc, Sc, Q1c, Q2c, mc)[evc] + 0.05)
    mu["shen_chao"] = shen_chao(Uf, Sf, Q1f, Q2f, mf) + 0.05
    bc, _ = bayes(cal_pre)
    phi["bayes"] = fit_phi(nc.values[evc], bc.expected_new(Z, Zd, Uc.values, mc.values)[evc])
    bf, _ = bayes(fwd_pre)
    mu["bayes"] = bf.expected_new(Z, Zd, Uf.values, mf.values)
    yv = nf.values[evf]
    ll = {k: nb_logpmf(yv, v[evf], phi[k]) for k, v in mu.items()}
    blk = (cxy[evf, 0] // 10) * 100000 + cxy[evf, 1] // 10
    ub = np.unique(blk)
    for k in ll:
        diff = ll[k] - ll["effort_glm"]
        bs = []
        for _ in range(500):
            pick = rng.choice(ub, len(ub))
            idx = np.concatenate([np.flatnonzero(blk == b_) for b_ in pick])
            bs.append(diff[idx].mean())
        rows.append({"test": "forward_discovery", "origin": T, "model": k, "n_cells": int(evf.sum()), "logscore": ll[k].mean(),
                     "delta_vs_effort_glm": diff.mean(), "ci_lo": np.quantile(bs, 0.025), "ci_hi": np.quantile(bs, 0.975)})
    print(T, {k: round(float(v.mean()), 3) for k, v in ll.items()}, flush=True)

Ua, Sa, Q1a, Q2a, inca = summary(2017, 2021)
po_sets = inca.groupby("c").speciesId.apply(set)
pa_sets = pa.groupby("c").speciesId.apply(set)
pa_n = pa.groupby("c").surveyId.nunique()
ba, _ = bayes((2017, 2021))
import torch
a_, b_ = torch.exp(torch.as_tensor(Zd, dtype=torch.float64, device="cuda") @ ba._split(ba.theta)[1]), \
         torch.exp(torch.as_tensor(Zd, dtype=torch.float64, device="cuda") @ ba._split(ba.theta)[2])
P0 = p_undetected(torch.as_tensor(Ua.values, dtype=torch.float64, device="cuda"), a_, b_).cpu().numpy()
with np.errstate(divide="ignore", invalid="ignore"):
    Uv, Q1v, Q2v = Ua.values, Q1a.values, Q2a.values
    covdef = np.nan_to_num(np.where(Uv > 0, Q1v / Uv * ((Uv - 1) * Q1v / ((Uv - 1) * Q1v + 2 * Q2v)), 1.0), nan=1.0)
tab = []
for c in pa_sets.index.intersection(pos.index):
    t = pa_sets[c]
    miss = len(t - po_sets.get(c, set())) / len(t)
    i = pos[c]
    tab.append({"c": c, "pa_surveys": pa_n[c], "pa_species": len(t), "frac_missed": miss, "po_visits": Uv[i],
                "bayes_P0": P0[i], "chao_coverage_deficit": covdef[i], "inv_visits": 1 / (1 + Uv[i])})
TT = pd.DataFrame(tab)
TT = TT[TT.po_visits > 0]
blk = ((TT.c // 100000) // 10) * 100000 + (TT.c % 100000) // 10
ub = np.unique(blk)
for pred in ("bayes_P0", "chao_coverage_deficit", "inv_visits"):
    r = spearmanr(TT[pred], TT.frac_missed)[0]
    bs = []
    for _ in range(500):
        pick = rng.choice(ub, len(ub))
        idx = np.concatenate([np.flatnonzero(blk.values == b_) for b_ in pick])
        bs.append(spearmanr(TT[pred].values[idx], TT.frac_missed.values[idx])[0])
    rows.append({"test": "pa_missed_fraction", "origin": 2021, "model": pred, "n_cells": len(TT), "logscore": np.nan,
                 "delta_vs_effort_glm": r, "ci_lo": np.nanquantile(bs, 0.025), "ci_hi": np.nanquantile(bs, 0.975)})
R = pd.DataFrame(rows)
R.to_csv(RESULTS / "E10_glc_discovery.csv", index=False)
print(R.round(3).to_string(index=False))
print("mean PA-confirmed fraction missed by PO:", round(TT.frac_missed.mean(), 3), "| mean P0:", round(TT.bayes_P0.mean(), 3))
