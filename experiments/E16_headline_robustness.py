"""E16: how robust is the E09b headline (bias-corrected SDMs find species iNaturalist never recorded)?

Runs from committed results only (no raw data needed):
  results/E09b_kba_unrecorded_pairs.csv.gz  per-pair scores and atlas truth
  results/E09_kba_cells.csv                 atlas lists and iNat visits per cell
  results/E15_endemic_gain_covariates.csv   endemic flag per species

1. Spatial-block bootstrap of the per-species AUC contrasts. Blocks of 0.3 deg and 0.5 deg are resampled
   jointly with species, so the CI covers both spatial autocorrelation and the species sample.
2. Atlas non-detection sensitivity:
   a. restrict truth to full-effort atlas cells (all 32 lists);
   b. tipping point: turn a fraction f of atlas absences into presences, chosen (i) at random and
      (ii) adversarially against the contrast (pairs the corrected model ranks low and the baseline ranks
      high), and find the f at which the advantage vanishes.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evst.paths import RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

SCORES = ["GLM_naive", "GLM_tgb", "BRT_naive", "BRT_tgb", "nearest_record"]
CONTRASTS = [("BRT_tgb", "nearest_record"), ("BRT_tgb", "BRT_naive"), ("GLM_tgb", "nearest_record"),
             ("GLM_tgb", "GLM_naive"), ("BRT_naive", "nearest_record")]

P = pd.read_csv(RESULTS / "E09b_kba_unrecorded_pairs.csv.gz")
C = pd.read_csv(RESULTS / "E09_kba_cells.csv").set_index("kcell")
endemic = pd.read_csv(RESULTS / "E15_endemic_gain_covariates.csv").set_index("species").endemic.astype(bool)
xy = P.kcell.str.strip("[]").str.split(",", expand=True).astype(float)
P["lon"], P["lat"] = xy[0], xy[1]
P["lists"] = C.kba_lists.reindex(P.kcell).to_numpy()
P["visits"] = C.inat_visits.reindex(P.kcell).to_numpy()


def eligible(df):
    n1 = df.groupby("species").found_by_kba.agg(["sum", "size"])
    return n1.index[(n1["sum"] >= 5) & (n1["sum"] <= n1["size"] - 5)]


def auc(y, s):
    """Mann-Whitney AUC (scores have no ties within species; checked)."""
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def per_species(df, scores=SCORES):
    sp = eligible(df)
    rows = [{"species": s, **{k: auc(g.found_by_kba.to_numpy(), g[k].to_numpy()) for k in scores}}
            for s, g in df[df.species.isin(sp)].groupby("species")]
    return pd.DataFrame(rows).set_index("species")


def summarise(T, label):
    out = []
    for m, r in CONTRASTS:
        d = (T[m] - T[r]).dropna()
        out.append({"analysis": label, "contrast": f"{m} - {r}", "n_species": len(d), "mean_delta": d.mean(),
                    "pct_better": (d > 0).mean(), "wilcoxon_p": wilcoxon(d).pvalue,
                    "auc_model": T[m].mean(), "auc_ref": T[r].mean()})
    return out


# ---------------------------------------------------------------- 1. spatial-block bootstrap
def weighted_auc_boot(y, s, W):
    """AUC under integer cell weights. y,s: (n,), W: (B, n). Returns (B,)."""
    o = np.argsort(s)
    y, W = y[o], W[:, o]
    wneg = W * (y == 0)
    below = np.cumsum(wneg, axis=1) - wneg  # negative weight strictly below each item
    num = (W * (y == 1) * below).sum(1)
    den = (W * (y == 1)).sum(1) * wneg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


rng = np.random.default_rng(a.seed)
sp_all = eligible(P)
base = per_species(P)
boot_rows, summary = [], summarise(base, "all pairs (E09b)")
for blk in (0.3, 0.5):
    bid = pd.factorize(np.floor(P.lon / blk).astype(int).astype(str) + "_" + np.floor(P.lat / blk).astype(int).astype(str))[0]
    nb = bid.max() + 1
    counts = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(a.boot)])  # (B, nb)
    A = {k: np.full((len(sp_all), a.boot), np.nan) for k in SCORES}
    for i, (s, g) in enumerate(P[P.species.isin(sp_all)].groupby("species")):
        W = counts[:, bid[g.index]].astype(np.float32)
        y = g.found_by_kba.to_numpy()
        for k in SCORES:
            A[k][i] = weighted_auc_boot(y, g[k].to_numpy(), W)
    spi = rng.integers(0, len(sp_all), (a.boot, len(sp_all)))  # resample species jointly
    is_end = endemic.reindex(sorted(sp_all)).fillna(False).to_numpy()
    for m, r in CONTRASTS:
        D = A[m] - A[r]  # (species, B)
        for subset, mask in (("all", np.ones(len(sp_all), bool)), ("endemic", is_end)):
            Ds = D[mask]
            sp_b = rng.integers(0, len(Ds), (a.boot, len(Ds)))
            means = np.array([np.nanmean(Ds[sp_b[b], b]) for b in range(a.boot)])
            pct = np.array([np.nanmean(Ds[sp_b[b], b] > 0) for b in range(a.boot)])
            point = (base[m] - base[r])[np.array(sorted(sp_all))[mask]]
            boot_rows.append({"block_deg": blk, "n_blocks": nb, "subset": subset, "contrast": f"{m} - {r}",
                              "n_species": int(mask.sum()), "mean_delta": point.mean(),
                              "ci_lo": np.nanpercentile(means, 2.5), "ci_hi": np.nanpercentile(means, 97.5),
                              "pct_better": (point > 0).mean(), "pct_ci_lo": np.nanpercentile(pct, 2.5),
                              "pct_ci_hi": np.nanpercentile(pct, 97.5), "p_boot_le0": np.mean(means <= 0)})
    del spi
B = pd.DataFrame(boot_rows)
B.to_csv(RESULTS / "E16_block_bootstrap.csv", index=False)
print("== 1. spatial-block bootstrap (blocks + species resampled jointly)")
print(B.round(4).to_string(index=False))

# ---------------------------------------------------------------- 2a. full-effort atlas cells
full = P[P.lists >= 32]
summary += summarise(per_species(full), f"atlas cells with 32 lists ({full.kcell.nunique()} cells)")
part = P[P.lists < 32]
summary += summarise(per_species(part), f"atlas cells with <32 lists ({part.kcell.nunique()} cells)")
S = pd.DataFrame(summary)
S.to_csv(RESULTS / "E16_effort_subsets.csv", index=False)
print("\n== 2a. effort-restricted truth")
print(S.round(4).to_string(index=False))

# ---------------------------------------------------------------- 2b. tipping point
Pe = P[P.species.isin(sp_all)].copy()
Pe["r_brt"] = Pe.groupby("species").BRT_tgb.rank(pct=True)
Pe["r_near"] = Pe.groupby("species").nearest_record.rank(pct=True)
Pe["r_naive"] = Pe.groupby("species").BRT_naive.rank(pct=True)
Pe["u"] = rng.random(len(Pe))
tip = []
for ref, rref in (("nearest_record", "r_near"), ("BRT_naive", "r_naive")):
    Pe["adv"] = Pe.r_brt - Pe[rref]  # most negative = worst for the corrected model
    for mode, key in (("random", "u"), ("adversarial", "adv"), ("low_BRT_tgb", "r_brt")):
        for f in (0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5):
            y = Pe.found_by_kba.to_numpy().copy()
            neg = Pe.found_by_kba == 0
            rk = Pe[neg].groupby("species")[key].rank(pct=True, method="first")
            flip = rk.index[rk <= f]
            y[Pe.index.get_indexer(flip)] = 1
            d = []
            for s, g in Pe.assign(y=y).groupby("species"):
                yy = g.y.to_numpy()
                if 5 <= yy.sum() <= len(yy) - 5:
                    d.append(auc(yy, g.BRT_tgb.to_numpy()) - auc(yy, g[ref].to_numpy()))
            d = np.array(d)
            tip.append({"ref": ref, "mode": mode, "frac_absences_flipped": f,
                        "implied_true_prevalence": Pe.found_by_kba.mean() + f * (1 - Pe.found_by_kba.mean()),
                        "mean_delta": d.mean(), "pct_better": (d > 0).mean()})
T = pd.DataFrame(tip)
T.to_csv(RESULTS / "E16_nondetection_tipping.csv", index=False)
print("\n== 2b. tipping point: BRT_tgb advantage after hiding presences among atlas absences")
print(T.pivot_table(index=["ref", "mode"], columns="frac_absences_flipped", values="mean_delta").round(3).to_string())
