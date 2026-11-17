import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from evst.paths import FIGURES, RESULTS

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
B = pd.read_csv(RESULTS / "E16_block_bootstrap.csv")
T = pd.read_csv(RESULTS / "E16_nondetection_tipping.csv")
S = pd.read_csv(RESULTS / "E17_summary.csv")
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.1), gridspec_kw={"width_ratios": [1.15, 1, 1]})

# A: block-bootstrap CIs
b = B[B.block_deg == 0.5]
order = ["BRT_tgb - nearest_record", "BRT_naive - nearest_record", "GLM_tgb - nearest_record",
         "BRT_tgb - BRT_naive", "GLM_tgb - GLM_naive"]
labels = ["Corrected BRT vs nearest record", "Uncorrected BRT vs nearest record", "Corrected GLM vs nearest record",
          "Correction gain, BRT", "Correction gain, GLM"]
for off, (sub, col, name) in zip((-0.14, 0.14), (("all", BLUE, "All 196 species"), ("endemic", ORANGE, "17 endemics"))):
    r = b[b.subset == sub].set_index("contrast").loc[order]
    y = np.arange(len(order)) + off
    ax[0].hlines(y, r.ci_lo, r.ci_hi, color=col, lw=2)
    ax[0].plot(r.mean_delta, y, "o", ms=6, color=col, mec="white", mew=1.5, label=name)
ax[0].axvline(0, color=GRAY, lw=1)
ax[0].set_yticks(range(len(order)), labels)
ax[0].invert_yaxis()
ax[0].set_xlabel("Δ mean per-species AUC on never-recorded pairs")
ax[0].set_title("A  Spatial-block bootstrap (0.5° blocks + species), 95% CI", loc="left", fontsize=9.5, color=INK)
ax[0].legend(frameon=False, loc="lower right")

# B: tipping point
t = T[T.ref == "nearest_record"]
for mode, col, name in (("random", BLUE, "at random"), ("low_BRT_tgb", ORANGE, "where corrected model ranks low"),
                        ("adversarial", AQUA, "worst case for the contrast")):
    r = t[t["mode"] == mode]
    ax[1].plot(100 * r.frac_absences_flipped, r.mean_delta, "-o", color=col, lw=2, ms=4, label=name)
ax[1].axhline(0, color=GRAY, lw=1)
ax[1].set_xlabel("% of atlas absences turned into hidden presences")
ax[1].set_ylabel("Corrected BRT − nearest record (AUC)")
ax[1].set_title("B  How many false atlas absences would erase it?", loc="left", fontsize=9.5, color=INK)
for x, y, txt, col in ((22, 0.072, "hidden at random", BLUE), (17, -0.035, "hidden where model ranks low", ORANGE),
                      (17, -0.17, "hidden adversarially", AQUA)):
    ax[1].text(x, y, txt, color=col, fontsize=8, va="bottom")

# C: environment-free recommenders, visited cells
s = S[S.stratum == "visited (all)"].set_index("method")
meth = ["BRT_tgb", "ensemble", "BRT_naive", "GLM_tgb", "nearest_record", "item_cf", "svd", "cell_knn"]
names = ["Corrected BRT", "BRT + item-CF ensemble", "Uncorrected BRT", "Corrected GLM", "Nearest record",
         "Item co-occurrence (CF)", "Matrix factorisation (SVD)", "Cell k-NN recommender"]
env = [True, True, True, True, False, False, False, False]
vals = s.loc[meth, "mean_auc"].to_numpy()
ax[2].barh(range(len(meth)), vals - 0.5, left=0.5, color=[BLUE if e else GRAY for e in env], height=0.62)
for i, v in enumerate(vals):
    ax[2].text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8, color=MUTED)
ax[2].set_yticks(range(len(meth)), names)
ax[2].invert_yaxis()
ax[2].set_xlim(0.5, 0.92)
ax[2].set_xlabel("Mean per-species AUC (194 species)")
ax[2].set_title("C  SDMs vs recommenders (visited cells)", loc="left", fontsize=9.5, color=INK)
ax[2].legend([Patch(color=BLUE), Patch(color=GRAY)], ["uses environment", "iNat records only"], frameon=False,
             loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "E16_E17_robustness.png", dpi=180)
print("saved", FIGURES / "E16_E17_robustness.png")
