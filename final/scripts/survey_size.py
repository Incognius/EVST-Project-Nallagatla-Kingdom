import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = next(p for p in (Path(__file__).resolve().parents[2] / "biodiv-gaps", Path(__file__).resolve().parents[2]) if (p / "src" / "evst").exists())
sys.path.insert(0, str(ROOT / "src"))
from evst.data import systems
from evst.paths import PROCESSED

OUT = Path(__file__).resolve().parents[1] / "results"
NS = [5, 10, 20, 40, 80, 160, 320]
DRAWS = 500
BLOCK = 0.3


def auc_matrix(S, Y, M):
    X = np.where(M, S, np.nan)
    r = rankdata(X, axis=0, nan_policy="omit")
    pos = M & (Y == 1)
    n1 = pos.sum(0)
    n0 = M.sum(0) - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (np.where(pos, r, 0).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    a[(n1 < 1) | (n0 < 1)] = np.nan
    return a


def evaluate(Sa, Sb, Y, M, idx, keep):
    a = auc_matrix(Sa[idx][:, keep], Y[idx][:, keep], M[idx][:, keep])
    b = auc_matrix(Sb[idx][:, keep], Y[idx][:, keep], M[idx][:, keep])
    return a - b


def run(name, Sys, sc, a, b, rng):
    Y, M, cxy = Sys["Y"], Sys["R"] == 0, Sys["cxy"]
    keep = np.flatnonzero([(5 <= Y[M[:, j], j].sum() <= M[:, j].sum() - 5) for j in range(Y.shape[1])])
    n_cells = len(cxy)
    full = evaluate(sc[a], sc[b], Y, M, np.arange(n_cells), keep)
    truth_sign = np.sign(full)
    blk = pd.factorize(np.floor(cxy[:, 0] / BLOCK) * 1000 + np.floor(cxy[:, 1] / BLOCK))[0]
    members = [np.flatnonzero(blk == u) for u in np.unique(blk)]
    rows = []
    for scheme in ("random cells", "clustered blocks"):
        for n in NS + [n_cells]:
            if n > n_cells:
                continue
            for d in range(DRAWS if n < n_cells else 1):
                if n == n_cells:
                    idx = np.arange(n_cells)
                elif scheme == "random cells":
                    idx = rng.choice(n_cells, n, replace=False)
                else:
                    order = rng.permutation(len(members))
                    idx = np.concatenate([members[o] for o in order])[:n]
                dl = evaluate(sc[a], sc[b], Y, M, idx, keep)
                ok = np.isfinite(dl)
                rows.append({"system": name, "scheme": scheme, "n_sites": n, "draw": d,
                             "species_scored": int(ok.sum()), "species_total": len(keep),
                             "mean_delta": float(np.nanmean(dl)) if ok.any() else np.nan,
                             "picks_corrected": bool(np.nanmean(dl) > 0) if ok.any() else np.nan,
                             "species_agree": float(np.mean(np.sign(dl[ok]) == truth_sign[ok])) if ok.any() else np.nan})
        print(name, scheme, "done", flush=True)
    return rows


t0 = time.time()
rng = np.random.default_rng(0)
rows = []
K = systems.get("kba")
rows += run("Kerala", K, dict(np.load(PROCESSED / "E16_brt_scores.npz")), "BRT_tgb", "BRT_naive", rng)
T = systems.get("taiwan")
rows += run("Taiwan", T, dict(np.load(PROCESSED / "E17_scores_taiwan.npz")), "BRT_tgb", "BRT_naive", rng)
D = pd.DataFrame(rows)
D.to_csv(OUT / "survey_size_draws.csv.gz", index=False)
S = (D.groupby(["system", "scheme", "n_sites"])
     .agg(p_pick_corrected=("picks_corrected", "mean"), species_agree=("species_agree", "mean"),
          species_scored=("species_scored", "median"), species_total=("species_total", "first"),
          mean_delta=("mean_delta", "mean"), delta_lo=("mean_delta", lambda v: np.nanquantile(v, 0.05)),
          delta_hi=("mean_delta", lambda v: np.nanquantile(v, 0.95)))
     .reset_index())
S.to_csv(OUT / "survey_size_summary.csv", index=False)
print(S.round(3).to_string(index=False))
print(f"{time.time() - t0:.0f}s")
