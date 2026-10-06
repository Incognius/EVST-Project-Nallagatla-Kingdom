import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evst.paths import FIGURES, RESULTS

REG = ["cv_rand_auc", "cv_spat_auc", "cv_spat_tg_auc"]
LAB = {"pa_auc": "independent PA (truth)", "cv_rand_auc": "random CV", "cv_spat_auc": "spatial-block CV",
       "cv_spat_tg_auc": "spatial CV vs target-group sites"}

df = pd.read_csv(RESULTS / "E02_disdat.csv")
regions = sorted(df.region.unique())
print("regions:", regions, "rows:", len(df), "species:", df.groupby("region").species.nunique().to_dict())

s = df.groupby(["model", "bg"])[["pa_auc"] + REG].mean().round(3)
s.to_csv(RESULTS / "E02_summary_model_bg.csv")
print(s)

w = df.pivot_table(index=["region", "species", "model"], columns="bg", values=["pa_auc"] + REG)
d = pd.DataFrame({k: w[(k, "tgb")] - w[(k, "random")] for k in ["pa_auc"] + REG}).dropna()
rows = []
for k in REG:
    agree = (np.sign(d[k]) == np.sign(d["pa_auc"])).mean()
    picks_tgb = (d[k] > 0).mean()
    rows.append({"regime": LAB[k], "agrees_with_truth_pct": 100 * agree, "says_tgb_better_pct": 100 * picks_tgb,
                 "spearman_delta_vs_truth": d[k].corr(d["pa_auc"], method="spearman")})
rows.append({"regime": LAB["pa_auc"], "agrees_with_truth_pct": 100.0, "says_tgb_better_pct": 100 * (d.pa_auc > 0).mean(),
             "spearman_delta_vs_truth": 1.0})
rev = pd.DataFrame(rows)
p = wilcoxon(d.pa_auc).pvalue
rev["n_pairs"] = len(d)
rev.to_csv(RESULTS / "E02_reversal.csv", index=False)
print(rev.round(3).to_string(), f"\nTGB vs random on PA truth: mean dAUC={d.pa_auc.mean():.3f}, Wilcoxon p={p:.2e}")

opt = df[df.bg == "random"].assign(**{f"opt_{k}": lambda x, k=k: x[k] - x.pa_auc for k in REG})
print(opt.groupby("region")[[f"opt_{k}" for k in REG]].mean().round(3))
opt.groupby("region")[[f"opt_{k}" for k in REG]].mean().to_csv(RESULTS / "E02_optimism_by_region.csv")

plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
fig, ax = plt.subplots(figsize=(6.4, 3.2))
keys = ["pa_auc"] + REG
x = np.arange(len(keys))
for i, (bg, col) in enumerate([("random", "#7f8c8d"), ("tgb", "#2e86c1")]):
    m = df[df.bg == bg][keys].mean()
    se = df[df.bg == bg][keys].sem()
    ax.bar(x + (i - 0.5) * 0.38, m, 0.38, yerr=se, color=col, label={"random": "uncorrected (random background)",
                                                                       "tgb": "target-group background (Phillips 2009)"}[bg])
ax.set_xticks(x, [LAB[k].replace(" vs ", "\nvs ") for k in keys], fontsize=8)
ax.set_ylim(0.5, 0.9)
ax.set_ylabel("mean AUC")
ax.legend(fontsize=7, loc="upper right")
ax.set_title(f"disdat: {df.species.nunique()} species, {len(regions)} regions, 4 algorithms")
fig.tight_layout()
fig.savefig(FIGURES / "E02_regimes.png")

fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
for a, k in zip(ax, REG):
    a.scatter(d[k], d.pa_auc, s=6, alpha=0.5, c="#2e86c1")
    a.axhline(0, c="k", lw=0.6)
    a.axvline(0, c="k", lw=0.6)
    agree = (np.sign(d[k]) == np.sign(d.pa_auc)).mean() * 100
    a.set_title(f"{LAB[k]}\nagrees with truth {agree:.0f}%", fontsize=8)
    a.set_xlabel("ΔAUC internal (TGB − random)")
ax[0].set_ylabel("ΔAUC on independent PA")
fig.tight_layout()
fig.savefig(FIGURES / "E02_reversal.png")
