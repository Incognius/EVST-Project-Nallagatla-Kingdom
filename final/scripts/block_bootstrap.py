import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = next(p for p in (Path(__file__).resolve().parents[2] / "biodiv-gaps", Path(__file__).resolve().parents[2]) if (p / "src" / "evst").exists())
sys.path.insert(0, str(ROOT / "src"))
from evst.data import systems
from evst.paths import PROCESSED

OUT = Path(__file__).resolve().parents[1] / "results"
B, BLOCK = 1000, 0.3


def auc_cols(y, S):
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 < 1 or n0 < 1:
        return np.full(S.shape[1], np.nan)
    r = np.apply_along_axis(rankdata, 0, S)
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def run(name, Sys, scores, masks, truths, pairs):
    cxy = Sys["cxy"]
    blk = pd.factorize(np.floor(cxy[:, 0] / BLOCK) * 1000 + np.floor(cxy[:, 1] / BLOCK))[0]
    ub = np.unique(blk)
    members = [np.flatnonzero(blk == b) for b in ub]
    keys = sorted({k for p in pairs for k in p})
    m = scores[keys[0]].shape[1]
    keep = [j for j in range(m) if 5 <= truths[masks[:, j], j].sum() <= masks[:, j].sum() - 5]

    def stat(idx):
        out = {p: [] for p in pairs}
        for j in keep:
            ii = idx[masks[idx, j]]
            y = truths[ii, j]
            a = auc_cols(y, np.column_stack([scores[k][ii, j] for k in keys]))
            d = dict(zip(keys, a))
            for p in pairs:
                out[p].append(d[p[0]] - d[p[1]])
        return {p: np.nanmean(v) for p, v in out.items()}

    full = stat(np.arange(len(cxy)))
    rng = np.random.default_rng(0)
    reps = {p: [] for p in pairs}
    for _ in range(B):
        pick = rng.integers(0, len(ub), len(ub))
        idx = np.concatenate([members[b] for b in pick])
        s = stat(idx)
        for p in pairs:
            reps[p].append(s[p])
    rows = []
    for p in pairs:
        r = np.array(reps[p])
        rows.append({"analysis": name, "contrast": f"{p[0]} - {p[1]}", "n_species": len(keep), "n_blocks": len(ub),
                     "delta": full[p], "ci_lo": np.nanquantile(r, 0.025), "ci_hi": np.nanquantile(r, 0.975),
                     "frac_reps_le0": float(np.mean(r <= 0))})
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
    return rows


rows = []
K = systems.get("kba")
sc = dict(np.load(PROCESSED / "E16_brt_scores.npz"))
rows += run("kerala_missing_pairs", K, sc, K["R"] == 0, K["Y"],
            [("BRT_tgb", "BRT_naive"), ("BRT_tgb", "nearest_record")])
T = systems.get("taiwan")
sc = dict(np.load(PROCESSED / "E17_scores_taiwan.npz"))
rows += run("taiwan_missing_pairs", T, sc, T["R"] == 0, T["Y"],
            [("BRT_tgb", "BRT_naive"), ("BRT_tgb", "nearest_record"), ("NN_occ_effort", "BRT_tgb")])
for sysname, end, f in (("kba", 2014, "E15_scores_2014"), ("kba", 2017, None), ("taiwan", 2016, "E15_scores_taiwan_2016"),
                        ("taiwan", 2018, "E15_scores_taiwan_2018")):
    if f is None:
        continue
    S = systems.get(sysname, end=end, min_cells=10)
    d = dict(np.load(PROCESSED / f"{f}.npz", allow_pickle=True))
    sc = {k: d[k] for k in ("BRT_naive", "BRT_tgb", "NN_an_ssdl", "NN_occ_effort")}
    eff = (S["R"] == 0) & (d["Ul"][:, None] > 0)
    pr = [("BRT_tgb", "BRT_naive"), ("NN_occ_effort", "NN_an_ssdl")]
    rows += run(f"flip_{sysname}_{end}_survey", S, sc, eff, S["Y"], pr)
    rows += run(f"flip_{sysname}_{end}_later_inat", S, sc, eff, d["later"], pr)
pd.DataFrame(rows).to_csv(OUT / "block_bootstrap_cis.csv", index=False)
