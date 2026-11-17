import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evst.paths import FIGURES, RESULTS

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
rows = []
for f in sorted(glob.glob(str(RESULTS / "E19" / "E19_*_*_*_s*.csv"))):
    model, bg, split, seed = f.split("/")[-1][:-4].split("_")[1:5]
    rows.append(pd.read_csv(f).assign(model=model, bg=bg, split=split, seed=int(seed[1:])))
A = pd.concat(rows)
A = A[A.n_pa_pres >= 10]
R = A[A.split == "random"]
per_seed = R.groupby(["model", "bg", "seed"])[["auc_pa", "excess_access_index"]].mean().reset_index()
M = R.groupby(["model", "bg", "species"]).mean(numeric_only=True).reset_index()
W = M.pivot_table(index="species", columns=["model", "bg"])
G = pd.read_csv(RESULTS / "E19" / "summary_cv_agreement.csv")

fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.1), gridspec_kw={"width_ratios": [1, 1.25, 1.15]})
combos = [("mlp", "random"), ("cnn", "random"), ("mlp", "tgb"), ("cnn", "tgb")]
labels = ["Point MLP\nrandom bg", "Patch CNN\nrandom bg", "Point MLP\ntarget-group", "Patch CNN\ntarget-group"]
cols = [GRAY, GRAY, BLUE, BLUE]

# A: survey AUC, seeds as dots
for i, (m, b) in enumerate(combos):
    v = per_seed[(per_seed.model == m) & (per_seed.bg == b)].auc_pa
    ax[0].bar(i, v.mean() - 0.7, bottom=0.7, color=cols[i], width=0.62, hatch="//" if m == "cnn" else None,
              edgecolor="white", lw=0)
    ax[0].plot(np.full(len(v), i), v, "o", ms=4, color=INK, alpha=0.6)
    ax[0].text(i, v.mean() + 0.004, f"{v.mean():.3f}", ha="center", fontsize=8, color=MUTED)
ax[0].set_xticks(range(4), labels, fontsize=8)
ax[0].set_ylim(0.7, 0.84)
ax[0].set_ylabel("Mean per-species AUC vs PA surveys")
ax[0].set_title(f"A  Skill on {int(M.species.nunique()):,} species' survey plots", loc="left", fontsize=9.5, color=INK)

# B: access excess per layer
layers = ["roads", "railways", "lights", "built", "popdensity"]
names = ["Roads", "Railways", "Night lights", "Built-up", "Population"]
x = np.arange(len(layers))
for k, ((m, b), lab, col) in enumerate(zip(combos, labels, cols)):
    v = [W[(f"excess_{l}", m, b)].mean() for l in layers]
    ax[1].bar(x + (k - 1.5) * 0.2, v, 0.19, color=col, hatch="//" if m == "cnn" else None, edgecolor="white",
              lw=0, label=lab.replace("\n", ", "))
ax[1].axhline(0, color=GRAY, lw=1)
ax[1].set_xticks(x, names)
ax[1].set_ylabel("Access excess (SD): top-k predicted − true sites")
ax[1].set_title("B  How far predictions drift towards people", loc="left", fontsize=9.5, color=INK)
ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.35)
ax[1].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")

# C: who picks the bias-corrected model?
g = G[G.choice.str.startswith("TGB vs random")]
checks = [("truth", "Survey truth"), ("auc_cv_vs_land", "Random CV"), ("spatial_block_cv_vs_land", "Spatial-block CV"),
          ("auc_cv_vs_tg", "Target-group CV")]
for k, (mm, col) in enumerate((("MLP", BLUE), ("CNN", ORANGE))):
    gg = g[g.choice.str.endswith(mm)]
    vals = []
    for c, _ in checks:
        if c == "truth":
            vals.append(gg.truth_prefers_first.iloc[0])
        else:
            s = gg[gg.check == c]
            vals.append(s.check_prefers_first.iloc[0] if len(s) else np.nan)
    ax[2].barh(np.arange(4) + (k - 0.5) * 0.36, np.array(vals) * 100, 0.34, color=col, label=mm)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax[2].text(v * 100 + 1, i + (k - 0.5) * 0.36, f"{v * 100:.0f}%", va="center", fontsize=7.5, color=MUTED)
ax[2].set_yticks(range(4), [n for _, n in checks])
ax[2].invert_yaxis()
ax[2].set_xlim(0, 112)
ax[2].set_xlabel("% of species where the check prefers the correction")
ax[2].set_title("C  Which check sees the benefit of correction?", loc="left", fontsize=9.5, color=INK)
ax[2].legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(FIGURES / "E19_glc_cnn.png", dpi=180)
print("saved", FIGURES / "E19_glc_cnn.png")
