from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import statsmodels.api as sm

from evst.bias.sampbias import fit_map, fit_nuts
from evst.data import records
from evst.data.rasters import ROAD100, load_stack
from evst.data.region import CRS, grid
from evst.paths import FIGURES, PROCESSED, RESULTS

FACTORS = ["d_road_major", "d_road_all", "d_town", "d_city", "d_river", "d_reserve"]
SOURCES = ["all", "eBird", "iNaturalist", "specimen", "other_observation"]
GROUPS = ["birds", "butterflies_moths", "odonates", "plants", "reptiles", "amphibians", "mammals"]


def units(df):
    src = [u for u in SOURCES if u == "all" or (df.source == u).sum() >= 1000]
    if len(src) <= 2:
        src = ["all"]
    return src + [f"group:{g}" for g in GROUPS if (df.group == g).sum() >= 1000]


def sel(df, src):
    if src == "all":
        return df
    if src.startswith("group:"):
        return df[df.group == src[6:]]
    return df[df.source == src]


def hughes(df, d100, tr100, layers, mask):
    rows = []
    r = ((df.y.values - tr100.f) / tr100.e).astype(int).clip(0, d100.shape[0] - 1)
    c = ((df.x.values - tr100.c) / tr100.a).astype(int).clip(0, d100.shape[1] - 1)
    df = df.assign(d_road=d100[r, c])
    m100 = np.repeat(np.repeat(mask, 10, 0), 10, 1)[: d100.shape[0], : d100.shape[1]]
    area_frac = float((d100[m100] <= 2.5).mean())
    n5 = len({(r // 5, c // 5) for r, c in zip(*np.nonzero(mask))})
    for src in [u for u in units(df) if not u.startswith("group:")]:
        for grp in ["all"] + sorted(df.group.unique()):
            s = sel(df, src)
            s = s if grp == "all" else s[s.group == grp]
            if len(s) < 200:
                continue
            spc = s.species.value_counts()
            top1 = max(1, int(np.ceil(0.01 * len(spc))))
            rows.append({"source": src, "group": grp, "records": len(s), "species": len(spc),
                         "pct_records_within_2.5km_road": 100 * (s.d_road <= 2.5).mean(),
                         "pct_area_within_2.5km_road": 100 * area_frac,
                         "pct_records_within_1km_road": 100 * (s.d_road <= 1).mean(),
                         "pct_0_1km": 100 * (s.d_road <= 1).mean(),
                         "pct_1_2.5km": 100 * ((s.d_road > 1) & (s.d_road <= 2.5)).mean(),
                         "pct_2.5_5km": 100 * ((s.d_road > 2.5) & (s.d_road <= 5)).mean(),
                         "pct_gt5km": 100 * (s.d_road > 5).mean(),
                         "pct_5km_cells_sampled": 100 * len(set(zip(s.row // 5, s.col // 5))) / n5,
                         "median_road_dist_km": s.d_road.median(),
                         "pct_1km_cells_sampled": 100 * s.cell.nunique() / mask.sum(),
                         "pct_records_top1pct_species": 100 * spc.iloc[:top1].sum() / len(s),
                         "pct_records_top100_species": 100 * spc.iloc[:100].sum() / len(s)})
    return pd.DataFrame(rows), df


def oliver(df, shape):
    c10 = (df.row // 10) * (shape[1] // 10 + 1) + df.col // 10
    df = df.assign(c10=c10).sort_values("year")
    rows = []
    for src in units(df):
        s = sel(df, src)
        seen_sc, seen_c = set(), set()
        for y, g in s.groupby("year"):
            if y < 1980:
                continue
            sc = set(zip(g.species, g.c10))
            new = len(sc - seen_sc)
            seen_sc |= sc
            seen_c |= set(g.cell)
            rows.append({"source": src, "year": int(y), "records": len(g), "new_species_cells": new,
                         "effectiveness_per_1000": 1000 * new / len(g), "cum_cells_1km": len(seen_c)})
    return pd.DataFrame(rows)


def cell_counts(df, src, n_cells, agg=5, shape=None):
    s = sel(df, src)
    r5, c5 = s.row // agg, s.col // agg
    return (r5 * (shape[1] // agg + 1) + c5).value_counts()


def agg_layer(a, mask, agg=5):
    h, w = a.shape
    H, W = int(np.ceil(h / agg)), int(np.ceil(w / agg))
    pad = np.full((H * agg, W * agg), np.nan, np.float32)
    pad[:h, :w] = np.where(mask, a, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(pad.reshape(H, agg, W, agg), axis=(1, 3))


def road_intensity(df, d100, mask):
    m100 = np.repeat(np.repeat(mask, 10, 0), 10, 1)[: d100.shape[0], : d100.shape[1]]
    bands = [0, 0.1, 0.25, 0.5, 1, 2, 5, 100]
    area = np.histogram(d100[m100], bins=bands)[0] * 0.01
    rows = []
    for g in ["all"] + GROUPS:
        s = df if g == "all" else df[df.group == g]
        n = np.histogram(s.d_road.values, bins=bands)[0]
        for i in range(len(area)):
            rows.append({"group": g, "band_km": f"{bands[i]}-{bands[i + 1]}", "area_km2": area[i], "records": n[i],
                         "pct_area": 100 * area[i] / area.sum(), "pct_records": 100 * n[i] / n.sum(),
                         "intensity_ratio": (n[i] / area[i]) / (n.sum() / area.sum())})
    return pd.DataFrame(rows)


def main(nuts: bool = True):
    df = records.load()
    layers, tr, mask = load_stack()
    tr_, shape, _ = grid()
    with rasterio.open(ROAD100) as src:
        d100, tr100 = src.read(1), src.transform

    h, df = hughes(df, d100, tr100, layers, mask)
    h.to_csv(RESULTS / "E01_hughes.csv", index=False)
    road_intensity(df, d100, mask).to_csv(RESULTS / "E01_road_intensity.csv", index=False)
    print(h[h.group == "all"].round(1).to_string())

    o = oliver(df, shape)
    o.to_csv(RESULTS / "E01_oliver.csv", index=False)

    agg = 5
    D5 = {f: agg_layer(layers[f], mask, agg) for f in FACTORS}
    valid = ~np.isnan(np.stack(list(D5.values()))).any(0)
    W5 = valid.shape[1]
    D = np.column_stack([D5[f][valid] for f in FACTORS])
    flat_ids = np.flatnonzero(valid.ravel())
    sb_rows = []
    for src in units(df):
        s = sel(df, src)
        ids = (s.row // agg) * W5 + s.col // agg
        counts = np.bincount(ids, minlength=valid.size)[flat_ids]
        m = fit_map(counts, D, FACTORS)
        row = {"source": src, "records": int(counts.sum()), "q": m["q"],
               **{f"w_{k}": v for k, v in m["w"].items()},
               **{f"halfdist_km_{k}": np.log(2) / v if v > 1e-9 else np.inf for k, v in m["w"].items()}}
        if nuts:
            n = fit_nuts(counts, D, FACTORS, draws=600, tune=600)
            row.update({f"w_{k}_lo": n["w_q025"][k] for k in FACTORS})
            row.update({f"w_{k}_hi": n["w_q975"][k] for k in FACTORS})
        sb_rows.append(row)
        print(src, {k: round(v, 2) for k, v in m["w"].items()}, flush=True)
    sb = pd.DataFrame(sb_rows)
    sb.to_csv(RESULTS / "E01_sampbias.csv", index=False)
    pd.DataFrame(np.column_stack([D, np.bincount((df.row // agg) * W5 + df.col // agg,
                                                 minlength=valid.size)[flat_ids]]),
                 columns=FACTORS + ["count"]).to_csv(PROCESSED / "sampbias_grid_5km.csv", index=False)

    acc = ["d_road_all", "d_road_major", "d_town", "d_city", "d_reserve", "pop_log"]
    Xa = np.column_stack([np.log1p(layers[k][mask]) if k.startswith("d_") else layers[k][mask] for k in acc])
    y = np.bincount(df.cell, minlength=mask.size)[np.flatnonzero(mask.ravel())]
    X = sm.add_constant((Xa - Xa.mean(0)) / Xa.std(0))
    glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    eff = np.full(mask.shape, np.nan, np.float32)
    eff[mask] = glm.predict(X)
    with rasterio.open(PROCESSED / "effort_1km.tif", "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
                       count=1, dtype="float32", crs=CRS, transform=tr, nodata=np.nan, compress="deflate") as dst:
        dst.write(eff, 1)
    from scipy.stats import spearmanr
    rho = spearmanr(eff[mask], y)[0]
    json.dump({"coef": dict(zip(["const"] + acc, glm.params.tolist())), "spearman_obs_vs_fit": rho,
               "pseudo_r2_deviance": 1 - glm.deviance / glm.null_deviance},
              open(RESULTS / "E01_effort_glm.json", "w"), indent=2)
    print("effort GLM spearman", round(rho, 3))
    figures(df, h, o, sb, layers, mask, eff)


def figures(df, h, o, sb, layers, mask, eff):
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    fig, ax = plt.subplots(1, 3, figsize=(9, 6))
    dens = np.zeros(mask.shape)
    np.add.at(dens, (df.row.values, df.col.values), 1)
    land5 = agg_layer(mask.astype(float), np.ones_like(mask), 5)
    dens5 = agg_layer(dens, np.ones_like(mask), 5) * 25
    dens5 = np.where(land5 > 0.5, np.log10(dens5 + 1), np.nan)
    road5 = np.where(land5 > 0.5, agg_layer(layers["d_road_all"], mask, 5), np.nan)
    eff5 = np.where(land5 > 0.5, np.log10(agg_layer(eff, mask, 5) + 1e-3), np.nan)
    for a, img, t in zip(ax, [dens5, road5, eff5],
                         ["log10 records per 5 km cell", "mean distance to road (km)", "fitted effort (log10)"]):
        im = a.imshow(img, cmap="magma" if "records" in t or "effort" in t else "viridis")
        a.set_title(t)
        a.axis("off")
        plt.colorbar(im, ax=a, shrink=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "E01_maps.png")
    hh = h[h.group == "all"].set_index("source")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(hh.index, hh["pct_records_within_2.5km_road"], color="#c0392b", label="records")
    ax.axhline(hh["pct_area_within_2.5km_road"].iloc[0], color="k", ls="--", label="land area")
    ax.set_ylabel("% within 2.5 km of a road")
    ax.legend()
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "E01_roads.png")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for src, g in o.groupby("source"):
        if src == "other_observation":
            continue
        ax[0].plot(g.year, g.records, label=src)
        ax[1].plot(g.year, g.effectiveness_per_1000.rolling(3, min_periods=1).mean(), label=src)
    ax[0].set_yscale("log")
    ax[0].set_title("records per year")
    ax[1].set_title("new species x 10 km cells per 1000 records")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "E01_oliver.png")


if __name__ == "__main__":
    import sys
    main(nuts="--no-nuts" not in sys.argv)
