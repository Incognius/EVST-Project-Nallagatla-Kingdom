"""Summarise E19/E19b: spatial context vs point models on GLC24 survey truth, bias leakage, and CV-vs-truth."""
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evst.paths import RESULTS

D = RESULTS / "E19"
rows = []
for f in sorted(glob.glob(str(D / "E19_*_*_*_s*.csv"))):
    model, bg, split, seed = f.split("/")[-1][:-4].split("_")[1:5]
    rows.append(pd.read_csv(f).assign(model=model, bg=bg, split=split, seed=int(seed[1:])))
A = pd.concat(rows)
A = A[A.n_pa_pres >= 10]
ex = [c for c in A.columns if c.startswith("excess_")]
metrics = ["auc_pa", "auc_cv_vs_land", "auc_cv_vs_tg"] + ex
# average seeds within species, random split = main analysis
M = A[A.split == "random"].groupby(["model", "bg", "species"])[metrics].mean().reset_index()
print("runs:", A.groupby(["model", "bg", "split"]).seed.nunique().to_dict())
summ = M.groupby(["model", "bg"])[metrics].mean()
print("\n== mean over species (random split, seeds averaged)\n", summ.round(4).T.to_string())

W = M.pivot_table(index="species", columns=["model", "bg"], values=metrics)
out = []


def contrast(label, a, b, metric):
    d = (W[(metric,) + a] - W[(metric,) + b]).dropna()
    out.append({"contrast": label, "metric": metric, "n_species": len(d), "mean_delta": d.mean(),
                "median_delta": d.median(), "pct_positive": (d > 0).mean(), "wilcoxon_p": wilcoxon(d).pvalue})


for metric in ("auc_pa", "excess_access_index", "excess_roads", "excess_popdensity", "auc_cv_vs_land", "auc_cv_vs_tg"):
    contrast("CNN - MLP (random bg)", ("cnn", "random"), ("mlp", "random"), metric)
    contrast("CNN - MLP (TGB)", ("cnn", "tgb"), ("mlp", "tgb"), metric)
    contrast("TGB - random (MLP)", ("mlp", "tgb"), ("mlp", "random"), metric)
    contrast("TGB - random (CNN)", ("cnn", "tgb"), ("cnn", "random"), metric)
C = pd.DataFrame(out)
print("\n== paired contrasts\n", C.round(4).to_string(index=False))

# which model would each check pick, per species? (H3 on real data with a CNN)
agree = []
for label, a, b in (("TGB vs random, MLP", ("mlp", "tgb"), ("mlp", "random")),
                    ("TGB vs random, CNN", ("cnn", "tgb"), ("cnn", "random")),
                    ("CNN vs MLP, random bg", ("cnn", "random"), ("mlp", "random")),
                    ("CNN vs MLP, TGB", ("cnn", "tgb"), ("mlp", "tgb"))):
    truth = W[("auc_pa",) + a] - W[("auc_pa",) + b]
    for check in ("auc_cv_vs_land", "auc_cv_vs_tg"):
        cv = W[(check,) + a] - W[(check,) + b]
        ok = truth.notna() & cv.notna()
        agree.append({"choice": label, "check": check, "n": int(ok.sum()), "truth_prefers_first": (truth[ok] > 0).mean(),
                      "check_prefers_first": (cv[ok] > 0).mean(),
                      "agreement": (np.sign(truth[ok]) == np.sign(cv[ok])).mean(),
                      "spearman_delta": pd.Series(truth[ok]).corr(pd.Series(cv[ok]), method="spearman")})
G = pd.DataFrame(agree)
print("\n== does the internal check pick the model the survey truth picks?\n", G.round(3).to_string(index=False))

# block split: same contrasts on PA truth (robustness) and spatial-CV agreement
if (A.split == "block").any():
    Bk = A[A.split == "block"].groupby(["model", "bg", "species"])[metrics].mean().reset_index()
    print("\n== block split, mean over species\n", Bk.groupby(["model", "bg"])[metrics].mean().round(4).T.to_string())
    Wb = Bk.pivot_table(index="species", columns=["model", "bg"], values=metrics)
    for label, a, b in (("TGB vs random, MLP", ("mlp", "tgb"), ("mlp", "random")), ("CNN vs MLP, random bg", ("cnn", "random"), ("mlp", "random"))):
        t = W[("auc_pa",) + a] - W[("auc_pa",) + b]
        cv = Wb[("auc_cv_vs_land",) + a] - Wb[("auc_cv_vs_land",) + b]
        ok = t.notna() & cv.notna()
        agree.append({"choice": label, "check": "spatial_block_cv_vs_land", "n": int(ok.sum()),
                      "truth_prefers_first": (t[ok] > 0).mean(), "check_prefers_first": (cv[ok] > 0).mean(),
                      "agreement": (np.sign(t[ok]) == np.sign(cv[ok])).mean(),
                      "spearman_delta": pd.Series(t[ok]).corr(pd.Series(cv[ok]), method="spearman")})
    G = pd.DataFrame(agree)
    print(G.tail(2).round(3).to_string(index=False))

brt = D / "E19b_brt.csv"
if brt.exists():
    Bt = pd.read_csv(brt)
    print("\n== BRT reference (subset of species)\n", Bt.drop(columns="species").mean().round(4).to_string())
    sub = M[M.species.isin(Bt.species)].groupby(["model", "bg"]).auc_pa.mean()
    print("NN PA AUC on the same species subset:\n", sub.round(4).to_string())
summ.to_csv(D / "summary_means.csv")
C.to_csv(D / "summary_contrasts.csv", index=False)
G.to_csv(D / "summary_cv_agreement.csv", index=False)
