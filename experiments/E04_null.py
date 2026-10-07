import numpy as np
import pandas as pd
import rasterio
import statsmodels.api as sm

from evst.data import records
from evst.data.rasters import load_stack
from evst.paths import PROCESSED, RESULTS

TRAIN_END, AGG = 2018, 5

layers, tr, mask = load_stack()
H, W = mask.shape
Hn, Wn = int(np.ceil(H / AGG)), int(np.ceil(W / AGG))
land = np.flatnonzero(mask.ravel())
r, c = np.divmod(land, W)
c5_of_land = (r // AGG) * Wn + (c // AGG)
land5 = np.bincount(c5_of_land, minlength=Hn * Wn) >= 13

df = records.load()
df = df.assign(c5=(df.row // AGG) * Wn + df.col // AGG)
early, late = df[df.year <= TRAIN_END], df[df.year > TRAIN_END]
obs_rec = np.bincount(early.c5, minlength=Hn * Wn).astype(float)
late_rec = np.bincount(late.c5, minlength=Hn * Wn).astype(float)
seen = set(zip(early.species, early.c5))
new = late[[(s, k) not in seen for s, k in zip(late.species, late.c5)]].drop_duplicates(["species", "c5"])
new_sc = np.bincount(new.c5, minlength=Hn * Wn).astype(float)

low_eff = land5 & (obs_rec <= np.quantile(obs_rec[land5], 0.25))
visited = low_eff & (late_rec > 0)
rows = []
for mname in ("GLM", "BRT"):
    for corr in ("naive", "tgb"):
        with rasterio.open(PROCESSED / f"E04_richness_{mname}_{corr}.tif") as src:
            pr = src.read(1).ravel()
        q = np.nanquantile(pr[visited], [0.25, 0.75])
        hi, lo = visited & (pr >= q[1]), visited & (pr <= q[0])
        rate = lambda m: 100 * new_sc[m].sum() / late_rec[m].sum()
        per_cell = pd.Series(new_sc[visited] / late_rec[visited])
        rows.append({"model": mname, "correction": corr, "low_effort_cells_revisited": int(visited.sum()),
                     "new_per_100_late_top_quartile_pred": rate(hi),
                     "new_per_100_late_bottom_quartile_pred": rate(lo),
                     "ratio_top_vs_bottom": rate(hi) / rate(lo),
                     "spearman_pred_vs_new_rate": per_cell.corr(pd.Series(pr[visited]), method="spearman")})
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "E04_null_low_effort.csv", index=False)
print(out.round(3).to_string(index=False))

reg = []
for mname in ("GLM", "BRT"):
    for corr in ("naive", "tgb"):
        with rasterio.open(PROCESSED / f"E04_richness_{mname}_{corr}.tif") as src:
            pr = src.read(1).ravel()
        z = (pr[visited] - pr[visited].mean()) / pr[visited].std()
        X = sm.add_constant(np.column_stack([np.log(late_rec[visited]), z]))
        fit = sm.GLM(new_sc[visited], X, family=sm.families.NegativeBinomial(alpha=0.5)).fit()
        lo, hi = fit.conf_int()[2]
        reg.append({"model": mname, "correction": corr, "beta_pred_per_sd": fit.params[2], "ci_lo": lo, "ci_hi": hi,
                    "beta_log_late_records": fit.params[1], "p": fit.pvalues[2]})
reg = pd.DataFrame(reg)
reg.to_csv(RESULTS / "E04_null_regression.csv", index=False)
print(reg.round(3).to_string(index=False))
