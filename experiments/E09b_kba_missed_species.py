import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

from evst.data import records
from evst.data.rasters import load_stack
from evst.models.sdm import BRT, GLM
from evst.paths import RAW, RESULTS
from evst.validate.blocks import random_folds
from evst.validate.metrics import auc

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--end", type=int, default=2020)
ap.add_argument("--min-cells", type=int, default=20)
ap.add_argument("--jobs", type=int, default=10)
a = ap.parse_args()
K = RAW / "kba"
kba = pd.read_csv(K / "kba_data.csv")
names = pd.read_csv(K / "kba_names.csv")
spd = pd.read_csv(K / "kba_species_details.csv")
names = names[names.Action.fillna("") != "Remove"]
kba = kba.merge(names[["Common.Name", "Assigned.Name.for.Atlas"]], on="Common.Name").merge(
    spd[["Common.Name", "Scientific.Name"]], left_on="Assigned.Name.for.Atlas", right_on="Common.Name", suffixes=("", "_sp"))
kba = kba[kba["Cell.ID"].notna()]
xy = kba["Cell.ID"].str.strip("[]").str.split(",", expand=True).astype(float)
kba["clon"], kba["clat"] = xy[0], xy[1]
cells = kba.drop_duplicates("Cell.ID")[["Cell.ID", "clon", "clat"]].reset_index(drop=True)
half = 0.03
tree = cKDTree(cells[["clon", "clat"]].values)
truth = kba.groupby("Cell.ID")["Scientific.Name"].apply(set)

layers, tr, mask = load_stack()
H, W = mask.shape
rr, cc = np.nonzero(mask)
ux, uy = tr * (cc + 0.5, rr + 0.5)
plon, plat = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True).transform(ux, uy)
_, pi = tree.query(np.column_stack([plon, plat]))
inside = (np.abs(plon - cells.clon.values[pi]) <= half) & (np.abs(plat - cells.clat.values[pi]) <= half)
pix = pd.DataFrame({"cell": rr * W + cc, "kcell": np.where(inside, cells["Cell.ID"].values[pi], None)})
cnt = pix.groupby("kcell").size()
ev = cnt[cnt >= 20].index.intersection(truth.index)
pix = pix[pix.kcell.isin(ev)]
ENV = [f"bio{i:02d}" for i in range(1, 20)] + ["elev", "wc_tree", "wc_shrub", "wc_grass", "wc_crop", "wc_built",
                                               "wc_bare", "wc_water", "wc_wetland", "wc_mangrove"]
land = np.flatnonzero(mask.ravel())
posl = pd.Series(np.arange(len(land)), index=land)
X = np.column_stack([layers[k].ravel()[land] for k in ENV])
pix = pix.assign(li=posl[pix.cell].to_numpy())

rec = records.load()
b = rec[(rec.group == "birds") & (rec.year <= a.end)]
b = b[b.cell.isin(posl.index)]
_, bi = tree.query(b[["decimallongitude", "decimallatitude"]].values)
ok = (np.abs(b.decimallongitude.values - cells.clon.values[bi]) <= half) & (np.abs(b.decimallatitude.values - cells.clat.values[bi]) <= half)
b = b.assign(kcell=np.where(ok, cells["Cell.ID"].values[bi], None))
inat_in_cell = b[b.kcell.notna()].groupby("kcell").species.apply(set)
matched = set(spd["Scientific.Name"]) & set(rec[rec.group == "birds"].species)
cellsets = b.groupby("species").cell.unique()
cand = sorted(s for s in matched if s in cellsets.index and len(cellsets[s]) >= a.min_cells)
tg = posl[b.cell.unique()].to_numpy()
rng = np.random.default_rng(0)
bg = rng.choice(len(land), 20000, replace=False)
bgte = rng.choice(len(land), 5000, replace=False)
cxy = cells.set_index("Cell.ID").loc[ev, ["clon", "clat"]].values
n_sampled_cells = b.cell.nunique()


def one(sp):
    pc = np.unique(posl[cellsets[sp]].to_numpy())
    preds, cv = {}, {}
    for mname, M in (("GLM", GLM), ("BRT", BRT)):
        for corr, bgc in (("naive", bg), ("tgb", tg)):
            bgc_ = np.setdiff1d(bgc, pc)
            m = M().fit(np.r_[X[pc], X[bgc_]], np.r_[np.ones(len(pc)), np.zeros(len(bgc_))])
            preds[f"{mname}_{corr}"] = pd.Series(m.predict(X[pix.li.to_numpy()]), index=pix.kcell.to_numpy()).groupby(level=0).mean().reindex(ev).to_numpy()
            if mname == "BRT":
                folds = random_folds(len(pc), 5, 1)
                aucs = []
                for k in range(5):
                    mk = M().fit(np.r_[X[pc[folds != k]], X[bgc_]], np.r_[np.ones((folds != k).sum()), np.zeros(len(bgc_))])
                    sp_, sb_ = mk.predict(X[pc[folds == k]]), mk.predict(X[bgte])
                    aucs.append(auc(np.r_[np.ones(len(sp_)), np.zeros(len(sb_))], np.r_[sp_, sb_]))
                cv[corr] = float(np.mean(aucs))
    sxy = b[b.species == sp][["decimallongitude", "decimallatitude"]].values
    dist = cKDTree(sxy).query(cxy)[0]
    preds["nearest_record"] = -dist
    preds["prevalence"] = np.full(len(ev), len(np.unique(cellsets[sp])) / n_sampled_cells)
    return sp, preds, cv


out = Parallel(n_jobs=a.jobs)(delayed(one)(sp) for sp in cand)
rows, cvrows, sp_rows = [], [], []
for sp, preds, cv in out:
    y = np.array([sp in truth[k] for k in ev], int)
    unrec = np.array([sp not in inat_in_cell.get(k, set()) for k in ev])
    if y.sum() >= 5:
        cvrows.append({"species": sp, "cv_naive": cv["naive"], "cv_tgb": cv["tgb"],
                       "kba_naive": auc(y, preds["BRT_naive"]), "kba_tgb": auc(y, preds["BRT_tgb"])})
    for k_i, k in enumerate(ev):
        if unrec[k_i]:
            rows.append({"species": sp, "kcell": k, "found_by_kba": y[k_i], **{n: v[k_i] for n, v in preds.items()}})
P = pd.DataFrame(rows)
P.to_csv(RESULTS / "E09b_kba_unrecorded_pairs.csv.gz", index=False, compression="gzip")
scores = ["GLM_naive", "GLM_tgb", "BRT_naive", "BRT_tgb", "nearest_record", "prevalence"]
per_sp = []
for sp, g in P.groupby("species"):
    if 5 <= g.found_by_kba.sum() <= len(g) - 5:
        per_sp.append({"species": sp, **{s: auc(g.found_by_kba, g[s]) for s in scores}})
S = pd.DataFrame(per_sp)
S.to_csv(RESULTS / "E09b_kba_unrecorded_species_auc.csv", index=False)
print(f"unrecorded (cell, species) pairs: {len(P):,}; found by KBA: {P.found_by_kba.mean():.3f}; species evaluated: {len(S)}")
print("mean per-species AUC at finding unrecorded species:", S[scores].mean().round(4).to_dict())
for s in ("GLM_tgb", "BRT_tgb", "BRT_naive"):
    for ref in ("nearest_record", "BRT_naive"):
        if s == ref:
            continue
        d = (S[s] - S[ref]).dropna()
        print(f"{s} - {ref}: {d.mean():+.4f}, better {100*(d>0).mean():.0f}%, p={wilcoxon(d).pvalue:.1e}")
C = pd.DataFrame(cvrows)
C.to_csv(RESULTS / "E09b_kba_cv_vs_truth.csv", index=False)
dcv, dtr = C.cv_tgb - C.cv_naive, C.kba_tgb - C.kba_naive
print(f"\nrandom CV says TGB better: {100*(dcv>0).mean():.1f}% | KBA truth says TGB better: {100*(dtr>0).mean():.1f}% | "
      f"agreement {100*(np.sign(dcv)==np.sign(dtr)).mean():.1f}% | n={len(C)}")
