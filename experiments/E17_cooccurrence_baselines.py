"""E17: does an environment-free "AI" find the missing species as well as the bias-corrected SDM?

A reviewer's obvious objection to E09b: a recommender system that completes the iNaturalist cell x species
matrix (species that co-occur with what *was* recorded in a cell) may do as well as an SDM, with no
environmental data at all. This builds those baselines from iNaturalist records only (no atlas information)
and scores them on the same never-recorded (cell, species) pairs as E09b.

Co-occurrence needs something recorded in the cell, so the head-to-head is on cells iNaturalist visited.
Unvisited cells (52%) are reported separately: there the recommenders have no information.

Methods (hyperparameters fixed before looking at atlas truth; a small sensitivity grid is reported):
  item_cf    mean Jaccard similarity between species s and the species recorded in the cell
  cell_knn   cosine-weighted share of the k most similar cells (by recorded composition) where s was recorded
  svd        rank-r reconstruction of the recorded matrix
  ensemble   mean within-species percentile rank of BRT_tgb and item_cf (no weights fitted)
Runs from committed results: results/E09b_kba_unrecorded_pairs.csv.gz, results/E09_kba_cells.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evst.paths import RESULTS

P = pd.read_csv(RESULTS / "E09b_kba_unrecorded_pairs.csv.gz")
C = pd.read_csv(RESULTS / "E09_kba_cells.csv").set_index("kcell")
cells = C.index.to_numpy()
species = np.sort(P.species.unique())
cpos = pd.Series(np.arange(len(cells)), index=cells)
spos = pd.Series(np.arange(len(species)), index=species)

# recorded matrix: a (cell, species) pair is recorded by iNat (<= 2020) iff it is absent from the unrecorded-pairs file
R = np.ones((len(cells), len(species)), np.float32)
R[cpos[P.kcell].to_numpy(), spos[P.species].to_numpy()] = 0
visits = C.inat_visits.to_numpy()
assert R[visits == 0].sum() == 0, "an unvisited cell has recorded species"
print(f"recorded matrix {R.shape}, recorded pairs {int(R.sum())}, visited cells {(visits > 0).sum()}, "
      f"visited cells with >=1 candidate species {(R.sum(1) > 0).sum()}")


def item_cf(R):
    inter = R.T @ R
    n = R.sum(0)
    J = inter / np.maximum(n[:, None] + n[None, :] - inter, 1)
    np.fill_diagonal(J, 0)
    k = R.sum(1, keepdims=True)
    return np.where(k > 0, (R @ J) / np.maximum(k, 1), np.nan)


def cell_knn(R, k):
    norm = np.linalg.norm(R, axis=1, keepdims=True)
    Z = R / np.maximum(norm, 1e-9)
    Sim = Z @ Z.T
    np.fill_diagonal(Sim, -np.inf)
    Sim[:, norm[:, 0] == 0] = -np.inf  # only borrow from cells with records
    idx = np.argsort(-Sim, axis=1)[:, :k]
    w = np.clip(np.take_along_axis(Sim, idx, 1), 0, None)
    out = (w[:, :, None] * R[idx]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9)
    out[norm[:, 0] == 0] = np.nan
    return out


def svd(R, r):
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    out = (U[:, :r] * s[:r]) @ Vt[:r]
    out[R.sum(1) == 0] = np.nan
    return out


PRIMARY = {"item_cf": item_cf(R), "cell_knn": cell_knn(R, 25), "svd": svd(R, 8)}
GRID = {"cell_knn_k10": cell_knn(R, 10), "cell_knn_k50": cell_knn(R, 50), "svd_r4": svd(R, 4), "svd_r16": svd(R, 16)}
ci, si = cpos[P.kcell].to_numpy(), spos[P.species].to_numpy()
for k, M in {**PRIMARY, **GRID}.items():
    P[k] = M[ci, si]
P["visits"] = visits[ci]
P["n_recorded_in_cell"] = R.sum(1)[ci]
# a little random jitter breaks ties (identical cells give identical recommender scores)
rng = np.random.default_rng(0)
for k in list(PRIMARY) + list(GRID):
    P[k] = P[k] + rng.random(len(P)) * 1e-9
P["ensemble"] = P.groupby("species").BRT_tgb.rank(pct=True) + P.groupby("species").item_cf.rank(pct=True)


def auc(y, s):
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


SC = ["BRT_tgb", "BRT_naive", "GLM_tgb", "nearest_record", "item_cf", "cell_knn", "svd", "ensemble"] + list(GRID)
strata = {"visited (all)": P.n_recorded_in_cell > 0,
          "visited 1-5 visits": (P.n_recorded_in_cell > 0) & (P.visits <= 5),
          "visited >5 visits": (P.n_recorded_in_cell > 0) & (P.visits > 5)}
rows = []
for name, m in strata.items():
    for sp, g in P[m].groupby("species"):
        y = g.found_by_kba.to_numpy()
        if 5 <= y.sum() <= len(y) - 5:
            rows.append({"stratum": name, "species": sp, "n_pairs": len(y), "prev": y.mean(),
                         **{k: auc(y, g[k].to_numpy()) for k in SC}})
T = pd.DataFrame(rows)
T.to_csv(RESULTS / "E17_species_auc.csv", index=False)

summ = []
for name, g in T.groupby("stratum", sort=False):
    for k in SC:
        d = g[k] - g["BRT_tgb"]
        summ.append({"stratum": name, "method": k, "n_species": len(g), "mean_auc": g[k].mean(),
                     "delta_vs_BRT_tgb": d.mean(), "pct_better_than_BRT_tgb": (d > 0).mean(),
                     "wilcoxon_p": wilcoxon(d).pvalue if k != "BRT_tgb" else np.nan})
S = pd.DataFrame(summ)
S.to_csv(RESULTS / "E17_summary.csv", index=False)
print(S.round(4).to_string(index=False))

# top-10% hit rate within visited cells (per species, then pooled), as in E09b
V = P[strata["visited (all)"]]
hit = {k: V.groupby("species", group_keys=False)
          .apply(lambda g: g.nlargest(max(1, round(0.1 * len(g))), k)).found_by_kba.mean() for k in SC[:8]}
print("\nvisited cells: base rate", round(V.found_by_kba.mean(), 3), "| top-10% hit rate",
      {k: round(v, 3) for k, v in hit.items()})
pd.Series(hit, name="top10_hit_rate").to_csv(RESULTS / "E17_top10_hit_visited.csv")
