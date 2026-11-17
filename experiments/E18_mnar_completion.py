import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from evst.data import systems
from evst.models import mnar
from evst.paths import PROCESSED, RESULTS

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--system", default="kba")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--skip-synthetic", action="store_true")
a = ap.parse_args()
suf = systems.suffix(a.system)
t0 = time.time()
S = systems.get(a.system)
Ut = mnar.units(S)
n_ev = Ut["n_ev"]
assert (Ut["R"][:n_ev] == S["R"]).all()
print(f"{Ut['n']} units, {n_ev} truth units, {int((Ut['U'] > 0).sum())} visited", flush=True)
dev = "cuda"


def ens(kind, T, p_fixed=None, seeds=a.seeds):
    acc, ex = 0, []
    for s in range(seeds):
        psi, e = mnar.fit(kind, T, seed=s, steps=a.steps, p_fixed=p_fixed)
        acc = acc + psi / psi.mean(0, keepdims=True)
        ex.append(e)
    return acc / seeds, ex


def run_all(T, Rn, Un):
    out, extras = {}, {}
    out["knn"] = mnar.knn(Rn, Un)
    for kind in mnar.KINDS:
        if kind == "sar_pu":
            continue
        out[kind], extras[kind] = ens(kind, T)
        print(kind, round(time.time() - t0), "s", flush=True)
    p = torch.as_tensor(np.mean([e["det"] for e in extras["occ_unit"]], 0), device=dev)
    out["sar_pu"], extras["sar_pu"] = ens("sar_pu", T, p_fixed=p)
    return out, extras


if not a.skip_synthetic:
    rng = np.random.default_rng(1)
    E, Acc, U = Ut["E"], Ut["Acc"], Ut["U"]
    m = 100
    W = rng.normal(0, 1, (E.shape[1], m)) / np.sqrt(E.shape[1]) * 1.5
    L = rng.normal(0, 1, (Ut["n"], 4)) @ rng.normal(0, 0.8, (4, m))
    zl = E @ W + L + rng.normal(-1.2, 0.6, m)
    z = rng.random(zl.shape) < 1 / (1 + np.exp(-zl))
    g_true = np.zeros(Acc.shape[1])
    g_true[0], g_true[4] = -0.6, 0.4
    pl = rng.normal(-1.5, 0.7, m)[None, :] + (Acc @ g_true)[:, None]
    mu = 1 - (1 - 1 / (1 + np.exp(-pl))) ** U[:, None]
    y = (z & (rng.random(z.shape) < mu)).astype(np.float32)
    Ts = mnar.tensors(dict(Ut, R=y))
    sc, ex = run_all(Ts, y, U)
    rows = []
    for k, v in sc.items():
        au = [roc_auc_score(z[:n_ev, j][y[:n_ev, j] == 0], v[:n_ev, j][y[:n_ev, j] == 0]) for j in range(m)
              if 5 <= z[:n_ev, j][y[:n_ev, j] == 0].sum() <= (y[:n_ev, j] == 0).sum() - 5]
        cos = np.nan
        if k == "expo_em":
            gh = np.mean([e["eg"] for e in ex[k]], 0)
            cos = gh @ g_true / (np.linalg.norm(gh) * np.linalg.norm(g_true) + 1e-9)
        rows.append({"model": k, "auc_hidden_presences": np.mean(au), "n_species": len(au), "exposure_cosine": cos})
    G = pd.DataFrame(rows)
    G.to_csv(RESULTS / f"E18_synthetic_gate{suf}.csv", index=False)
    print(G.round(4).to_string(index=False), flush=True)

T = mnar.tensors(Ut)
sc, ex = run_all(T, Ut["R"], Ut["U"])
np.savez_compressed(PROCESSED / f"E18_unit_scores{suf}.npz", **{k: v[:n_ev] for k, v in sc.items()})
gh = np.mean([e["eg"] for e in ex["expo_em"]], 0)
pd.DataFrame({"access_layer": ["d_road_all", "d_road_major", "d_town", "d_reserve", "pop_log"], "exposure_coef": gh}).to_csv(
    RESULTS / f"E18_exposure_coefs{suf}.csv", index=False)
ref = {}
if a.system == "kba":
    B = np.load(PROCESSED / "E16_brt_scores.npz")
    ref = {"BRT_tgb": B["BRT_tgb"], "BRT_naive": B["BRT_naive"], "nearest_record": B["nearest_record"],
           "occ_effort_pix": np.load(PROCESSED / "E11_cell_scores_occ_effort.npy")}
    pfa = np.load(PROCESSED / "E16_atlas_occupancy.npz")["pfa"]
else:
    B = np.load(PROCESSED / f"E17_scores{suf}.npz")
    ref = {k: B[k] for k in ("BRT_tgb", "BRT_naive", "nearest_record", "NN_occ_effort") if k in B}
    pfa = None
allsc = {**{k: v[:n_ev] for k, v in sc.items()}, **ref}
Y, R, Uc = S["Y"], S["R"], Ut["U"][:n_ev]
rows = []
for j, sp in enumerate(S["cand"]):
    for stratum, m in (("all_missing", R[:, j] == 0), ("missing_unvisited", (R[:, j] == 0) & (Uc == 0)),
                       ("missing_visited", (R[:, j] == 0) & (Uc > 0))):
        y = Y[m, j]
        if 5 <= y.sum() <= len(y) - 5:
            rows.append({"species": sp, "stratum": stratum, **{k: roc_auc_score(y, v[m, j]) for k, v in allsc.items()}})
D = pd.DataFrame(rows)
D.to_csv(RESULTS / f"E18_species_auc{suf}.csv", index=False)
models = list(allsc)
M = D.groupby("stratum")[models].mean().T
print(M.round(4).to_string(), flush=True)
cs = []
pairs = [("knn", "mf"), ("env_mf_tg", "env_tg"), ("expo_em", "env_mf_tg"), ("expo_em", "occ_unit"), ("sar_pu", "env_mf_tg"),
         ("expo_em", "expo_em_noacc"), ("occ_unit", "env_tg"), ("expo_em", "BRT_tgb"), ("env_mf_tg", "BRT_tgb"),
         ("knn", "BRT_tgb"), ("mf", "BRT_tgb")]
for st, g in D.groupby("stratum"):
    for x, r in pairs:
        if x in g and r in g:
            d = g[x] - g[r]
            cs.append({"stratum": st, "contrast": f"{x} - {r}", "n_species": len(g), "delta": d.mean(),
                       "pct_better": (d > 0).mean(), "wilcoxon_p": wilcoxon(d).pvalue})
C = pd.DataFrame(cs)
C.to_csv(RESULTS / f"E18_contrasts{suf}.csv", index=False)
print(C.round(4).to_string(index=False))
print("exposure coefs", dict(zip(["d_road_all", "d_road_major", "d_town", "d_reserve", "pop_log"], gh.round(3))))
print("total", round(time.time() - t0), "s")
