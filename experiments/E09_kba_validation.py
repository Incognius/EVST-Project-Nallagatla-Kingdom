import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from evst.bias.discovery import DiscoveryModel
from evst.data import records
from evst.data.rasters import load_stack
from evst.models.sdm import BRT, GLM
from evst.paths import RAW, RESULTS
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
kba = kba.merge(names[["Common.Name", "Assigned.Name.for.Atlas"]], on="Common.Name", how="inner")
kba = kba.merge(spd[["Common.Name", "Scientific.Name", "Endemicity"]], left_on="Assigned.Name.for.Atlas",
                right_on="Common.Name", how="inner", suffixes=("", "_sp"))
kba = kba[kba["Cell.ID"].notna()]
xy = kba["Cell.ID"].str.strip("[]").str.split(",", expand=True).astype(float)
kba["clon"], kba["clat"] = xy[0], xy[1]
cells = kba.drop_duplicates("Cell.ID")[["Cell.ID", "clon", "clat"]].reset_index(drop=True)
lons, lats = np.sort(cells.clon.unique()), np.sort(cells.clat.unique())
half = 0.5 * min(np.median(np.diff(lons)), np.median(np.diff(lats)))
tree = cKDTree(cells[["clon", "clat"]].values)
truth = kba.groupby("Cell.ID")["Scientific.Name"].apply(set)
n_lists = kba.groupby("Cell.ID")["List.ID"].nunique()

rec = records.load()
birds = rec[(rec.group == "birds") & (rec.year <= a.end)].copy()
dd, ii = tree.query(birds[["decimallongitude", "decimallatitude"]].values)
dl = np.abs(birds.decimallongitude.values - cells.clon.values[ii])
dt = np.abs(birds.decimallatitude.values - cells.clat.values[ii])
birds = birds[(dl <= half) & (dt <= half)].assign(kcell=cells["Cell.ID"].values[ii[(dl <= half) & (dt <= half)]])
birds["visit"] = birds.recordedby.astype(str) + "|" + pd.to_datetime(birds.eventdate).dt.normalize().astype(str)
kba_sp = set(spd["Scientific.Name"])
inat_sp = set(rec[rec.group == "birds"].species)
matched = kba_sp & inat_sp
print(f"half-cell {half:.4f} deg; KBA species {len(kba_sp)}, iNat bird species {len(inat_sp)}, matched {len(matched)}")

layers, tr, mask = load_stack()
H, W = mask.shape
rr, cc = np.nonzero(mask)
ux, uy = tr * (cc + 0.5, rr + 0.5)
plon, plat = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True).transform(ux, uy)
pd_, pi = tree.query(np.column_stack([plon, plat]))
inside = (np.abs(plon - cells.clon.values[pi]) <= half) & (np.abs(plat - cells.clat.values[pi]) <= half)
pix = pd.DataFrame({"cell": rr * W + cc, "kcell": np.where(inside, cells["Cell.ID"].values[pi], None)})
ENV = [f"bio{i:02d}" for i in range(1, 20)] + ["elev", "wc_tree", "wc_shrub", "wc_grass", "wc_crop", "wc_built",
                                               "wc_bare", "wc_water", "wc_wetland", "wc_mangrove"]
ACC = ["d_road_all", "d_road_major", "d_town", "d_reserve", "pop_log"]
flat = {k: layers[k][rr, cc] for k in ENV + ACC}
pix = pix.assign(**flat)
kc = pix[pix.kcell.notna()].groupby("kcell")
cov = kc[ENV + ACC].mean()
cov["n_pix"] = kc.size()
ev = cells.set_index("Cell.ID").join(cov, how="inner")
ev = ev[(ev.n_pix >= 20) & ev.index.isin(truth.index)]
print("KBA cells inside study region with >=20 land pixels:", len(ev))

g = birds[birds.kcell.isin(ev.index)]
U = g.groupby("kcell").visit.nunique().reindex(ev.index, fill_value=0).astype(float)
inc = g.groupby(["kcell", "species"]).visit.nunique().rename("Y").reset_index()
S = inc.groupby("kcell").size().reindex(ev.index, fill_value=0).astype(float)
Q1 = inc[inc.Y == 1].groupby("kcell").size().reindex(ev.index, fill_value=0).astype(float)
Q2 = inc[inc.Y == 2].groupby("kcell").size().reindex(ev.index, fill_value=0).astype(float)
inat_sets = g.groupby("kcell").species.apply(set)
pos = pd.Series(np.arange(len(ev)), index=ev.index)
Ez = PCA(8, random_state=0).fit_transform(((ev[ENV] - ev[ENV].mean()) / ev[ENV].std()).fillna(0).values)
Az = ((ev[ACC] - ev[ACC].mean()) / ev[ACC].std()).fillna(0).values
rng = np.random.default_rng(0)
Om, ph = rng.normal(0, 1 / 0.6, (2, 24)), rng.uniform(0, 2 * np.pi, 24)
RFF = np.sqrt(2 / 24) * np.cos(ev[["clon", "clat"]].values @ Om + ph)
ONE = np.ones((len(ev), 1))
ci, Yv = pos[inc.kcell].to_numpy(), inc.Y.to_numpy(float)
bayes = {}
for name, (Z, Zd) in {"bayes_full": (np.hstack([ONE, Ez, RFF]), np.hstack([ONE, Az])), "bayes_intercepts": (ONE, ONE)}.items():
    m = DiscoveryModel().fit(Z, Zd, U.values, S.values, ci, Yv)
    import torch
    th = m.theta
    wL, wa, wb = m._split(th)
    t = lambda x: torch.as_tensor(np.asarray(x, np.float64), device=m.device)
    a_, b_ = torch.exp(t(Zd) @ wa), torch.exp(t(Zd) @ wb)
    from evst.bias.discovery import p_undetected
    P0 = p_undetected(t(U.values), a_, b_).cpu().numpy()
    Lam = torch.exp(t(Z) @ wL).cpu().numpy()
    bayes[name] = (P0, Lam * P0)

with np.errstate(divide="ignore", invalid="ignore"):
    Uv, Q1v, Q2v, Sv = U.values, Q1.values, Q2.values, S.values
    cov_def = np.where(Uv > 0, Q1v / Uv * ((Uv - 1) * Q1v / ((Uv - 1) * Q1v + 2 * Q2v)), 1.0)
    cov_def = np.nan_to_num(cov_def, nan=1.0)
    f0 = np.where(Q2v > 0, (Uv - 1) / Uv * Q1v ** 2 / (2 * Q2v), (Uv - 1) / Uv * Q1v * np.maximum(Q1v - 1, 0) / 2)
    f0 = np.nan_to_num(np.where(Uv > 0, f0, 0.0))

rows = []
for kcell in ev.index:
    tset = truth[kcell] & matched
    iset = inat_sets.get(kcell, set())
    missed = tset - iset
    rows.append({"kcell": kcell, "kba_species": len(tset), "kba_lists": int(n_lists.get(kcell, 0)), "inat_visits": U[kcell],
                 "inat_species": S[kcell], "missed": len(missed), "frac_missed": len(missed) / max(len(tset), 1),
                 "missed_endemic": len({s for s in missed if s in set(spd[spd.Endemicity != 'Not endemic']['Scientific.Name'])})})
T = pd.DataFrame(rows).set_index("kcell")
T["bayes_full_P0"], T["bayes_full_undetected"] = bayes["bayes_full"]
T["bayes_icpt_P0"], T["bayes_icpt_undetected"] = bayes["bayes_intercepts"]
T["chao_coverage_deficit"], T["chao2_f0"] = cov_def, f0
T["inv_visits"] = 1 / (1 + T.inat_visits)


def spatial_boot_spearman(x, y, blocks, n=1000):
    ub = np.unique(blocks)
    out = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub))
        idx = np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        out.append(spearmanr(x[idx], y[idx])[0])
    return np.nanquantile(out, [0.025, 0.975])


sampled = T[T.inat_visits > 0]
blocks = (np.floor(ev.loc[sampled.index, "clon"] / 0.3) * 1000 + np.floor(ev.loc[sampled.index, "clat"] / 0.3)).values
res = []
for target in ("frac_missed", "missed"):
    for pred in ("bayes_full_P0", "bayes_icpt_P0", "chao_coverage_deficit", "inv_visits", "bayes_full_undetected",
                 "bayes_icpt_undetected", "chao2_f0"):
        r = spearmanr(sampled[pred], sampled[target])[0]
        lo, hi = spatial_boot_spearman(sampled[pred].values, sampled[target].values, blocks)
        res.append({"target": target, "predictor": pred, "spearman": r, "ci_lo": lo, "ci_hi": hi, "n_cells": len(sampled)})
R1 = pd.DataFrame(res)
T.to_csv(RESULTS / "E09_kba_cells.csv")
R1.to_csv(RESULTS / "E09_kba_missed_species.csv", index=False)
print(R1.round(3).to_string(index=False))
print("mean observed frac missed (sampled cells):", round(sampled.frac_missed.mean(), 3),
      "| mean predicted P0 full:", round(sampled.bayes_full_P0.mean(), 3), "icpt:", round(sampled.bayes_icpt_P0.mean(), 3),
      "| cells never visited by iNat:", int((T.inat_visits == 0).sum()))

land = np.flatnonzero(mask.ravel())
posl = pd.Series(np.arange(len(land)), index=land)
X = np.column_stack([layers[k].ravel()[land] for k in ENV])
b1 = birds.copy()
b1 = rec[(rec.group == "birds") & (rec.year <= a.end)]
b1 = b1[b1.cell.isin(posl.index)]
tg = posl[b1.cell.unique()].to_numpy()
cellsets = b1.groupby("species").cell.unique()
cand = [s for s in matched if s in cellsets.index and len(cellsets[s]) >= a.min_cells]
bg = rng.choice(len(land), 20000, replace=False)
kpix = pix[pix.kcell.isin(ev.index)]
kpix = kpix.assign(li=posl[kpix.cell].to_numpy())
surveyed = ev.index[ev.index.isin(truth.index)]


def one(sp):
    pc = np.unique(posl[cellsets[sp]].to_numpy())
    y = np.array([sp in truth[k] for k in surveyed], int)
    if y.sum() < 5 or y.sum() > len(y) - 5:
        return []
    out = []
    for mname, M in (("GLM", GLM), ("BRT", BRT)):
        for corr, bgc in (("naive", bg), ("tgb", tg)):
            bgc = np.setdiff1d(bgc, pc)
            m = M().fit(np.r_[X[pc], X[bgc]], np.r_[np.ones(len(pc)), np.zeros(len(bgc))])
            pr = pd.Series(m.predict(X[kpix.li.to_numpy()]), index=kpix.kcell.to_numpy()).groupby(level=0).mean()
            out.append({"species": sp, "model": mname, "correction": corr, "n_train_cells": len(pc),
                        "kba_prevalence": y.mean(), "auc_vs_kba": auc(y, pr.reindex(surveyed).to_numpy())})
    return out


rows2 = [r for o in Parallel(n_jobs=a.jobs)(delayed(one)(sp) for sp in sorted(cand)) for r in o]
R2 = pd.DataFrame(rows2)
R2.to_csv(RESULTS / "E09_kba_sdm_auc.csv", index=False)
w = R2.pivot_table(index=["species", "model"], columns="correction", values="auc_vs_kba").dropna()
from scipy.stats import wilcoxon
print("\nSDM vs KBA truth:", R2.groupby(["model", "correction"]).auc_vs_kba.mean().round(4).to_dict())
for mname in ("GLM", "BRT"):
    d_ = w.xs(mname, level="model")
    dd_ = d_["tgb"] - d_["naive"]
    print(f"{mname}: tgb - naive = {dd_.mean():+.4f}, tgb better {100*(dd_>0).mean():.0f}% of {len(dd_)} species, wilcoxon p={wilcoxon(dd_).pvalue:.2e}")
