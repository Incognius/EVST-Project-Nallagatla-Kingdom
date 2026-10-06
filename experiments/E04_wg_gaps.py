from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed

from evst.data import records
from evst.data.rasters import load_stack
from evst.models.sdm import BRT, GLM
from evst.paths import FIGURES, PROCESSED, RESULTS
from evst.validate.metrics import auc

warnings.filterwarnings("ignore")
ENV = [f"bio{i:02d}" for i in range(1, 20)] + ["elev", "wc_tree", "wc_shrub", "wc_grass", "wc_crop", "wc_built",
                                               "wc_bare", "wc_water", "wc_wetland", "wc_mangrove"]
GROUPS = ["birds", "amphibians", "reptiles", "butterflies_moths", "odonates", "mammals"]
MODELS = {"GLM": GLM, "BRT": BRT}


def fit_species(sp, X, pres_cells, bg_cells, tg_cells, cand_new, cand_neg, models, rng_seed=0):
    out, preds = [], {}
    rng = np.random.default_rng(rng_seed)
    for mname in models:
        for corr, bgc in (("naive", bg_cells), ("tgb", tg_cells)):
            bgc = np.setdiff1d(bgc, pres_cells)
            if len(bgc) > 20000:
                bgc = rng.choice(bgc, 20000, replace=False)
            m = MODELS[mname]().fit(np.r_[X[pres_cells], X[bgc]], np.r_[np.ones(len(pres_cells)), np.zeros(len(bgc))])
            p_all = m.predict(X)
            thr = np.quantile(p_all[pres_cells], 0.10)
            preds[(mname, corr)] = (p_all >= thr)
            da = auc(np.r_[np.ones(len(cand_new)), np.zeros(len(cand_neg))],
                     np.r_[p_all[cand_new], p_all[cand_neg]]) if len(cand_new) >= 3 else np.nan
            out.append({"species": sp, "model": mname, "correction": corr, "n_train_cells": len(pres_cells),
                        "n_new_cells": len(cand_new), "n_surveyed_neg": len(cand_neg), "discovery_auc": da,
                        "pred_area_frac": float(preds[(mname, corr)].mean())})
    return out, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", type=int, default=2018)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--max-species", type=int, default=400)
    ap.add_argument("--models", default="GLM,BRT")
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    layers, tr, mask = load_stack()
    land = np.flatnonzero(mask.ravel())
    pos = -np.ones(mask.size, np.int64)
    pos[land] = np.arange(len(land))
    X = np.column_stack([layers[k].ravel()[land] for k in ENV])
    df = records.load()
    df = df[pos[df.cell.values] >= 0].assign(li=lambda d: pos[d.cell.values])
    early, late = df[df.year <= a.train_end], df[df.year > a.train_end]
    rng = np.random.default_rng(0)
    bg_cells = rng.choice(len(land), 20000, replace=False)
    disc_rows, rich = [], {}
    H, W = mask.shape
    for grp in GROUPS:
        e, l = early[early.group == grp], late[late.group == grp]
        tg_cells = np.unique(e.li.values)
        late_tg = np.unique(l.li.values)
        cnt = e.groupby("species").li.nunique()
        sps = cnt[cnt >= a.min_cells].sort_values(ascending=False).index[: a.max_species]
        if len(sps) == 0:
            continue
        e_cells = e.groupby("species").li.unique()
        l_cells = l.groupby("species").li.unique()
        jobs = []
        for sp in sps:
            pc = np.unique(e_cells[sp])
            new = np.setdiff1d(np.unique(l_cells.get(sp, np.array([], int))), pc)
            neg = np.setdiff1d(np.setdiff1d(late_tg, pc), new)
            jobs.append(delayed(fit_species)(sp, X, pc, bg_cells, tg_cells, new, neg, a.models.split(",")))
        res = Parallel(n_jobs=a.jobs)(jobs)
        for (rows, preds), sp in zip(res, sps):
            for r in rows:
                r["group"] = grp
            disc_rows += rows
            for k, v in preds.items():
                rich.setdefault((grp,) + k, np.zeros(len(land), np.int32))
                rich[(grp,) + k] += v
        d = pd.DataFrame(disc_rows)
        print(grp, len(sps), "species;", d[d.group == grp].groupby(["model", "correction"]).discovery_auc.mean().round(3).to_dict(),
              flush=True)
        d.to_csv(RESULTS / "E04_discovery.csv", index=False)

    agg = 5
    Hn, Wn = int(np.ceil(H / agg)), int(np.ceil(W / agg))
    r, c = np.divmod(land, W)
    c5 = (r // agg) * Wn + (c // agg)
    obs_rec = np.bincount(c5[early.li.values], minlength=Hn * Wn).astype(float)
    obs_sp = early.assign(c5=c5[early.li.values]).groupby("c5").species.nunique().reindex(range(Hn * Wn), fill_value=0).values
    land5 = np.bincount(c5, minlength=Hn * Wn) >= 13
    later_new = late.assign(c5=c5[late.li.values])
    first_seen = early.assign(c5=c5[early.li.values])[["species", "c5"]].drop_duplicates()
    ln = later_new[["species", "c5"]].drop_duplicates().merge(first_seen, how="left", indicator=True)
    new_sc = ln[ln["_merge"] == "left_only"].groupby("c5").size().reindex(range(Hn * Wn), fill_value=0).values
    late_rec = np.bincount(later_new.c5.values, minlength=Hn * Wn).astype(float)
    summ = []
    for mname in a.models.split(","):
        for corr in ("naive", "tgb"):
            tot = sum(v for k, v in rich.items() if k[1] == mname and k[2] == corr)
            pr5 = np.bincount(c5, weights=tot, minlength=Hn * Wn) / np.maximum(np.bincount(c5, minlength=Hn * Wn), 1)
            ok = land5
            hi_pred = pr5 >= np.quantile(pr5[ok], 0.75)
            lo_eff = obs_rec <= np.quantile(obs_rec[ok], 0.25)
            gap = ok & hi_pred & lo_eff
            from scipy.stats import spearmanr
            summ.append({"model": mname, "correction": corr, "n_gap_cells": int(gap.sum()),
                         "rho_predrich_vs_records": spearmanr(pr5[ok], obs_rec[ok])[0],
                         "rho_predrich_vs_obsrich": spearmanr(pr5[ok], obs_sp[ok])[0],
                         "new_per_100_late_records_gap": 100 * new_sc[gap].sum() / max(late_rec[gap].sum(), 1),
                         "new_per_100_late_records_other": 100 * new_sc[ok & ~gap].sum() / max(late_rec[ok & ~gap].sum(), 1),
                         "late_records_in_gap_cells": int(late_rec[gap].sum())})
            img = np.full(Hn * Wn, np.nan)
            img[ok] = pr5[ok]
            prof = dict(driver="GTiff", height=Hn, width=Wn, count=2, dtype="float32", crs="EPSG:32643",
                        transform=tr * tr.scale(agg, agg), nodata=np.nan, compress="deflate")
            with rasterio.open(PROCESSED / f"E04_richness_{mname}_{corr}.tif", "w", **prof) as dst:
                dst.write(img.reshape(Hn, Wn).astype("float32"), 1)
                dst.write(np.where(ok, gap, np.nan).reshape(Hn, Wn).astype("float32"), 2)
    s = pd.DataFrame(summ)
    s.to_csv(RESULTS / "E04_gaps_summary.csv", index=False)
    print(s.round(3).to_string())
    figs(Hn, Wn, land5, obs_rec, a.models.split(",")[0])


def figs(Hn, Wn, land5, obs_rec, mname):
    plt.rcParams.update({"figure.dpi": 150, "font.size": 8})
    fig, ax = plt.subplots(1, 4, figsize=(10, 5.5))
    rec = np.where(land5, np.log10(obs_rec + 1), np.nan).reshape(Hn, Wn)
    ax[0].imshow(rec, cmap="magma")
    ax[0].set_title("records ≤2018 (log10)")
    for a_, corr in zip(ax[1:3], ("naive", "tgb")):
        with rasterio.open(PROCESSED / f"E04_richness_{mname}_{corr}.tif") as s:
            a_.imshow(s.read(1), cmap="viridis")
        a_.set_title(f"predicted richness\n{mname} {corr}")
    with rasterio.open(PROCESSED / f"E04_richness_{mname}_tgb.tif") as s:
        g = s.read(2)
    ax[3].imshow(np.where(land5.reshape(Hn, Wn), 0.3, np.nan), cmap="Greys", vmin=0, vmax=1)
    ax[3].imshow(np.where(g == 1, 1, np.nan), cmap="autumn")
    ax[3].set_title("gap cells (tgb):\nhigh predicted, low recorded")
    for a_ in ax:
        a_.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "E04_gaps.png")


if __name__ == "__main__":
    main()
