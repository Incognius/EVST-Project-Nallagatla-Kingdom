"""E23: how many independent survey sites does it take to choose the right model?

H3 shows that random CV on citizen-science records prefers the uncorrected model for almost every species, however many
records it uses. The practical follow-up: if validation needs independent survey data, how much is enough?

Design (per system, per replicate):
  1. Split the survey sites into two spatially disjoint halves by block (Kerala: 0.3 deg blocks; GLC24: 50 km blocks).
     Half A is the reference truth. Half B is the pool a hypothetical survey is drawn from. Because the sample never
     touches the reference, agreement is with independent truth, not with itself.
  2. From B draw a survey of n sites, either (i) at random or (ii) clustered: whole blocks in random order until n sites
     are reached (a realistic field campaign).
  3. On the sample, compute each species' survey AUC for both models of a contrast and the per-species difference.
  4. Score the decision the sample implies against the reference half:
       - project-level: sign of the mean per-species difference (pick one method for all species), and whether a
         one-sample Wilcoxon test on the differences is significant (p < 0.05) in the right / wrong direction;
       - species-level: share of species whose sign matches the reference (species with >= 2 presences and >= 2 absences
         in the sample, >= 5 presences in the reference).
  The same reference halves also score random CV's species-level choices (CV uses no survey data at all), so the two
  are compared on identical truth.

Kerala runs on the committed pairs file (unrecorded pairs only, the H5 protocol). GLC24 needs survey-plot predictions
saved by `E19_glc_cnn.py --save-pa` (run on Ada with scripts/ada_e23.sh).
Usage: python experiments/E23_survey_budget.py --system kerala|glc [--reps N]
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from evst.paths import INTERIM, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--system", choices=["kerala", "glc"], required=True)
ap.add_argument("--reps", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
t0 = time.time()
(RESULTS / "E23").mkdir(exist_ok=True)


# ---------------------------------------------------------------- per-species AUC on a subset of sites
def auc_masked(Y, P, M):
    """Per-column AUC over rows where M is True. Y, M bool (n, S); P float (n, S). NaN where undefined.
    Ineligible rows get -inf, so they rank below every eligible row and are subtracted out exactly."""
    Pm = np.where(M, P, -np.inf)
    R = rankdata(Pm, axis=0)  # average ranks: ties handled
    n_out = (~M).sum(0)
    pos = Y & M
    n1, n0 = pos.sum(0), (M & ~Y).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((R * pos).sum(0) - n1 * n_out - n1 * (n1 + 1) / 2) / (n1 * n0), n1, n0


if a.system == "glc":
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def auc_dense(Yt, Pt):
        """Per-column AUC on GPU, all rows eligible (scores are float32, ties negligible)."""
        R = torch.empty_like(Pt)
        R.scatter_(0, Pt.argsort(0), torch.arange(1, len(Pt) + 1, device=dev, dtype=Pt.dtype)
                   .view(-1, 1).expand_as(Pt).contiguous())
        n1 = Yt.sum(0)
        n0 = len(Yt) - n1
        return (((R * Yt).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)).cpu().numpy(), n1.cpu().numpy(), n0.cpu().numpy()

# ---------------------------------------------------------------- load a system as dense site x species arrays
if a.system == "kerala":
    D = pd.read_csv(RESULTS / "E09b_kba_unrecorded_pairs.csv.gz")
    sites = np.sort(D.kcell.unique())
    sp = np.sort(D.species.unique())
    i = np.searchsorted(sites, D.kcell)
    j = np.searchsorted(sp, D.species)
    N, S = len(sites), len(sp)
    M = np.zeros((N, S), bool)
    M[i, j] = True
    Y = np.zeros((N, S), bool)
    Y[i, j] = D.found_by_kba.to_numpy() == 1
    models = {}
    for m in ["BRT_tgb", "BRT_naive", "GLM_tgb", "GLM_naive", "nearest_record"]:
        P = np.zeros((N, S))
        P[i, j] = D[m].to_numpy()
        models[m] = P
    xy = pd.Series(sites).str.strip("[]").str.split(",", expand=True).astype(float).to_numpy()
    block = (np.floor(xy[:, 0] / 0.3) * 1000 + np.floor(xy[:, 1] / 0.3)).astype(np.int64)
    contrasts = [("BRT_tgb", "BRT_naive", "Correction (BRT)"),
                 ("BRT_tgb", "nearest_record", "Corrected BRT vs nearest record"),
                 ("GLM_tgb", "GLM_naive", "Correction (GLM)")]
    grid = [10, 20, 30, 50, 75, 100, 150, 200, 300]
    reps = a.reps or 500
    # random CV's choice (BRT, 5-fold CV on iNaturalist records): uses no survey data
    cv = pd.read_csv(RESULTS / "E09b_kba_cv_vs_truth.csv").set_index("species")
    cv_choice = {"Correction (BRT)": {"random CV": np.sign((cv.cv_tgb - cv.cv_naive).reindex(sp).to_numpy())}}

    def auc_fn(idx, m):
        return auc_masked(Y[idx], models[m][idx], M[idx])
else:
    G = INTERIM / "glc24"
    meta = __import__("json").loads((G / "meta.json").read_text())
    W = meta["shape"][1]
    pa = np.load(G / "pa.npz")
    sp = np.load(G / "species.npy")
    Yn = pa["Y"].astype(bool)
    N, S = Yn.shape
    cell = pa["cell"]
    block = ((cell % W) // 50).astype(np.int64) * 100000 + (cell // W) // 50
    Yt = torch.tensor(Yn, dtype=torch.uint8, device=dev)  # 8-bit on the GPU; each sample is cast to float
    names = {"mlp_tgb": "E19_mlp_tgb_random_s0_pred", "mlp_random": "E19_mlp_random_random_s0_pred",
             "cnn_tgb": "E19_cnn_tgb_random_s0_pred", "cnn_random": "E19_cnn_random_random_s0_pred"}
    names = {k: v for k, v in names.items() if (G / "preds" / f"{v}.npy").exists()}
    models = {k: torch.tensor(np.load(G / "preds" / f"{v}.npy"), device=dev) for k, v in names.items()}
    contrasts = [c for c in [("mlp_tgb", "mlp_random", "Correction (MLP)"), ("cnn_tgb", "cnn_random", "Correction (CNN)")]
                 if c[0] in models and c[1] in models]
    grid = [25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    reps = a.reps or 200
    cv_choice = {}
    for first, second, label in contrasts:
        A_ = pd.read_csv(RESULTS / "E23" / f"{names[first]}.csv").set_index("species").reindex(sp)
        B_ = pd.read_csv(RESULTS / "E23" / f"{names[second]}.csv").set_index("species").reindex(sp)
        cv_choice[label] = {"random CV": np.sign((A_.auc_cv_vs_land - B_.auc_cv_vs_land).to_numpy()),
                            "target-group CV": np.sign((A_.auc_cv_vs_tg - B_.auc_cv_vs_tg).to_numpy())}

    def auc_fn(idx, m):
        it = torch.as_tensor(idx, device=dev)
        return auc_dense(Yt[it].float(), models[m][it])

print(f"{a.system}: {N:,} survey sites, {S} species, {len(np.unique(block))} blocks, models {list(models)}, "
      f"{reps} reps, load {time.time() - t0:.0f}s", flush=True)
ub = np.unique(block)
by_block = {b: np.flatnonzero(block == b) for b in ub}


def delta(idx, first, second, min_pos, min_neg=2):
    a1, n1, n0 = auc_fn(idx, first)
    a2, _, _ = auc_fn(idx, second)
    d = a1 - a2
    d[(n1 < min_pos) | (n0 < min_neg)] = np.nan
    return d


def draw(pool_blocks, n, design):
    if design == "random":
        pool = np.concatenate([by_block[b] for b in pool_blocks])
        return rng.choice(pool, min(n, len(pool)), replace=False)
    out, k = [], 0
    for b in rng.permutation(pool_blocks):
        out.append(by_block[b])
        k += len(by_block[b])
        if k >= n:
            break
    idx = np.concatenate(out)
    return idx[:n] if len(idx) > n else idx  # trim the last block so every sample has exactly n sites


rows, cvrows = [], []
for rep in range(reps):
    perm = rng.permutation(ub)
    A_blocks, B_blocks = perm[: len(ub) // 2], perm[len(ub) // 2:]
    A_idx = np.concatenate([by_block[b] for b in A_blocks])
    nB = sum(len(by_block[b]) for b in B_blocks)
    for first, second, label in contrasts:
        ref = delta(A_idx, first, second, min_pos=5)
        ref_ok = np.isfinite(ref) & (ref != 0)
        ref_mean = np.nanmean(ref[ref_ok])
        for src, choice in cv_choice.get(label, {}).items():
            ok = ref_ok & np.isfinite(choice) & (choice != 0)
            cvrows.append({"rep": rep, "contrast": label, "check": src, "n_species": ok.sum(),
                           "species_agree": np.mean(choice[ok] == np.sign(ref[ok])),
                           "check_prefers_first": np.mean(choice[ok] > 0), "ref_prefers_first": np.mean(ref[ok] > 0)})
        for design in ["random", "clustered"]:
            for n in grid + ["all"]:
                nn = nB if n == "all" else n
                if nn > nB:
                    continue
                d = delta(draw(B_blocks, nn, design), first, second, min_pos=2)
                ok = np.isfinite(d) & (d != 0)
                both = ok & ref_ok
                r = {"rep": rep, "contrast": label, "design": design, "n": n, "n_sites": nn,
                     "n_species": ok.sum(), "ref_mean": ref_mean, "ref_species": ref_ok.sum()}
                if ok.sum() >= 1:
                    m = d[ok].mean()
                    p = wilcoxon(d[ok]).pvalue if ok.sum() >= 6 else 1.0
                    r.update(mean_delta=m, p=p, correct=np.sign(m) == np.sign(ref_mean),
                             sig_correct=(p < 0.05) and np.sign(m) == np.sign(ref_mean),
                             sig_wrong=(p < 0.05) and np.sign(m) != np.sign(ref_mean))
                if both.sum():
                    r.update(species_agree=np.mean(np.sign(d[both]) == np.sign(ref[both])), n_species_both=both.sum())
                rows.append(r)
    if rep % max(1, reps // 10) == 0:
        print(f"rep {rep} {time.time() - t0:.0f}s", flush=True)

R = pd.DataFrame(rows)
R.to_csv(RESULTS / "E23" / f"E23_{a.system}_reps.csv.gz", index=False)
pd.DataFrame(cvrows).to_csv(RESULTS / "E23" / f"E23_{a.system}_cv_reps.csv", index=False)
for c in ["correct", "sig_correct", "sig_wrong"]:
    R[c] = R[c].astype(float)
summ = (R.groupby(["contrast", "design", "n"], sort=False)
        .agg(n_sites=("n_sites", "median"), reps=("rep", "size"), p_correct=("correct", "mean"),
             p_sig_correct=("sig_correct", "mean"), p_sig_wrong=("sig_wrong", "mean"),
             species_agree=("species_agree", "mean"), n_species=("n_species", "median"),
             n_species_both=("n_species_both", "median"), ref_mean=("ref_mean", "mean")).reset_index())
summ.to_csv(RESULTS / "E23" / f"E23_{a.system}_summary.csv", index=False)
CV = pd.DataFrame(cvrows)
if len(CV):
    cvs = CV.groupby(["contrast", "check"]).mean(numeric_only=True).drop(columns="rep").reset_index()
    cvs.to_csv(RESULTS / "E23" / f"E23_{a.system}_cv_summary.csv", index=False)
    print(cvs.to_string(index=False))
pd.set_option("display.width", 200)
print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"done {time.time() - t0:.0f}s")
