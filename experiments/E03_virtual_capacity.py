from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed

from evst.data.rasters import load_stack
from evst.models.batched import GPUStack, fit_predict
from evst.models.sdm import BRT, GLM, RFDown
from evst.paths import PROCESSED, RESULTS
from evst.validate.metrics import auc, leakage
from evst.virtual.species import make_species, sample_presences

warnings.filterwarnings("ignore")
ENV = [f"bio{i:02d}" for i in range(1, 20)] + ["elev"] + ["wc_tree", "wc_shrub", "wc_grass", "wc_crop", "wc_built",
                                                           "wc_bare", "wc_water", "wc_wetland", "wc_mangrove"]
TRUTH_VARS = [f"bio{i:02d}" for i in (1, 4, 5, 6, 12, 14, 15, 17)] + ["elev", "wc_tree", "wc_grass", "wc_shrub"]
ACC = ["d_road_all", "d_road_major", "d_town", "d_city", "pop_log"]


def load(effort="effort_1km.tif"):
    layers, tr, mask = load_stack()
    with rasterio.open(PROCESSED / effort) as src:
        eff = src.read(1)
    mask &= ~np.isnan(eff)
    cells = np.flatnonzero(mask.ravel())
    env = np.column_stack([layers[k].ravel()[cells] for k in ENV])
    acc = np.column_stack([(np.log1p(layers[k]) if k.startswith("d_") else layers[k]).ravel()[cells] for k in ACC])
    tv = np.column_stack([layers[k].ravel()[cells] for k in TRUTH_VARS])
    return layers, mask, cells, env, acc, tv, eff.ravel()[cells]


def tabular(model_cls, Xp, Xb, Xe, Xp_te=None, Xb_te=None):
    m = model_cls().fit(np.r_[Xp, Xb], np.r_[np.ones(len(Xp)), np.zeros(len(Xb))])
    pe = m.predict(Xe)
    cv = None
    if Xp_te is not None:
        cv = auc(np.r_[np.ones(len(Xp_te)), np.zeros(len(Xb_te))], np.r_[m.predict(Xp_te), m.predict(Xb_te)])
    return pe, cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-species", type=int, default=30)
    ap.add_argument("--n-pres", type=int, default=300)
    ap.add_argument("--strengths", default="0,1,2")
    ap.add_argument("--models", default="GLM,BRT,RF,MLP,CNN")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="E03_virtual.csv")
    ap.add_argument("--effort", default="effort_1km.tif")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    layers, mask, cells, env, acc, tv, eff = load(a.effort)
    N = len(cells)
    print(f"{N:,} land cells", flush=True)
    species = [make_species(tv, rng) for _ in range(a.n_species)]
    ev = rng.choice(N, 20000, replace=False)
    pa_real = [rng.random(len(ev)) < sp.suit[ev] for sp in species]
    bg_all = rng.choice(N, 10000, replace=False)
    bg_tr, bg_te = bg_all[:8000], bg_all[8000:]
    H, W = mask.shape
    chan = ENV + ACC
    full = np.stack([(np.log1p(layers[k]) if k.startswith("d_") else layers[k]) for k in chan]).astype(np.float32)
    full[:, ~mask] = np.nan
    gstack = GPUStack(full, patch=9)
    env_ch, all_ch = list(range(len(ENV))), list(range(len(chan)))
    acc_med_std = {len(ENV) + j: float((np.nanmedian(full[len(ENV) + j]) - gstack.mu[len(ENV) + j])
                                       / gstack.sd[len(ENV) + j]) for j in range(len(ACC))}
    acc_med = np.nanmedian(acc, 0)
    models = a.models.split(",")
    rows = []
    for s in [float(x) for x in a.strengths.split(",")]:
        t0 = time.time()
        pres = [sample_presences(sp, eff, s, a.n_pres, rng) for sp in species]
        split = [rng.permutation(len(p)) for p in pres]
        p_tr = [p[i[: int(0.8 * len(p))]] for p, i in zip(pres, split)]
        p_te = [p[i[int(0.8 * len(p)):]] for p, i in zip(pres, split)]
        tgb = np.unique(np.concatenate(pres))
        realised = float(np.mean([np.log(eff[p]).mean() for p in pres]) - np.log(eff[bg_all]).mean())
        W = mask.shape[1]
        rc = np.divmod(cells, W)
        cell10 = (rc[0] // 10) * (W // 10 + 1) + rc[1] // 10
        for corr in ("none", "tgb", "biascov", "thin"):
            bgc = tgb if corr == "tgb" else bg_tr
            if corr == "thin":
                p_use = [p[np.unique(cell10[p], return_index=True)[1]] for p in p_tr]
            else:
                p_use = p_tr
            X = np.c_[env, acc] if corr == "biascov" else env
            Xe = np.c_[env[ev], np.tile(acc_med, (len(ev), 1))] if corr == "biascov" else env[ev]
            preds, cvs = {}, {}
            for mname, cls in (("GLM", GLM), ("BRT", BRT), ("RF", RFDown)):
                if mname not in models:
                    continue
                out = Parallel(n_jobs=a.jobs)(delayed(tabular)(
                    cls, X[p_use[k]], X[bgc[~np.isin(bgc, p_te[k])]] if corr == "tgb" else X[bgc], Xe,
                    X[p_te[k]], np.c_[env[bg_te], acc[bg_te]] if corr == "biascov" else env[bg_te])
                    for k in range(len(species)))
                preds[mname] = np.array([o[0] for o in out])
                cvs[mname] = [o[1] for o in out]
            for kind in ("MLP", "CNN"):
                if kind not in models:
                    continue
                ch = all_ch if corr == "biascov" else env_ch
                bgl = [bgc[~np.isin(bgc, p_te[k])] for k in range(len(species))] if corr == "tgb" \
                    else [bgc] * len(species)
                cv_cells = np.unique(np.concatenate(p_te + [bg_te]))
                pe, pe_cv = fit_predict(kind.lower(), gstack, [cells[p] for p in p_use], [cells[b] for b in bgl],
                                        cells[ev], steps=a.steps, channels=ch, seed=a.seed,
                                        override=acc_med_std if corr == "biascov" else None,
                                        extra_cells=cells[cv_cells])
                preds[kind] = pe
                pos = {c: i for i, c in enumerate(cv_cells)}
                cvs[kind] = [auc(np.r_[np.ones(len(p_te[k])), np.zeros(len(bg_te))],
                                 np.r_[pe_cv[k, [pos[c] for c in p_te[k]]], pe_cv[k, [pos[c] for c in bg_te]]])
                             for k in range(len(species))]
            for mname, P in preds.items():
                for k, sp in enumerate(species):
                    rows.append({"strength": s, "realised_bias": realised, "correction": corr, "model": mname,
                                 "species": k, "prevalence": sp.prevalence,
                                 "true_auc": auc(pa_real[k], P[k]),
                                 "spearman": pd.Series(P[k]).corr(pd.Series(sp.suit[ev]), method="spearman"),
                                 "cv_auc": cvs[mname][k],
                                 "leakage": leakage(P[k], acc[ev], truth=np.log(sp.suit[ev] / (1 - sp.suit[ev])))})
            pd.DataFrame(rows).to_csv(RESULTS / a.out, index=False)
            print(f"s={s} {corr} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby(["strength", "correction", "model"])[["true_auc", "cv_auc", "leakage"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
