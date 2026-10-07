import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evst.paths import FIGURES, RESULTS

ORDER = ["GLM", "BRT", "RF", "MLP", "CNN"]
CORR = ["none", "thin", "biascov", "tgb"]
COL = {"GLM": "#7f8c8d", "BRT": "#e67e22", "RF": "#27ae60", "MLP": "#8e44ad", "CNN": "#2e86c1"}

df = pd.read_csv(RESULTS / "E03_virtual.csv")
s = df.groupby(["strength", "correction", "model"])[["true_auc", "spearman", "cv_auc", "leakage"]].mean()
s.round(3).to_csv(RESULTS / "E03_summary.csv")
print(s.round(3).unstack("model")["leakage"][ORDER].to_string())

base = df[df.strength == 0].groupby(["correction", "model", "species"]).leakage.mean()
df["excess_leakage"] = df.leakage - df.set_index(["correction", "model", "species"]).index.map(base).values
none2 = df[(df.strength == 2) & (df.correction == "none")]
cap = none2.groupby("model")[["true_auc", "cv_auc", "leakage", "excess_leakage"]].mean().loc[ORDER]
cap.round(3).to_csv(RESULTS / "E03_capacity_strong_bias.csv")
print("\nStrong bias, uncorrected:\n", cap.round(3).to_string())
glm = none2[none2.model == "GLM"].set_index("species").excess_leakage
tests = {m: wilcoxon(none2[none2.model == m].set_index("species").excess_leakage - glm).pvalue
         for m in ORDER if m != "GLM"}
print("Wilcoxon vs GLM (excess leakage):", {k: f"{v:.1e}" for k, v in tests.items()})

rows = []
for strength in sorted(df.strength.unique()):
    d = df[df.strength == strength]
    for m in ORDER:
        dm = d[d.model == m].pivot_table(index="species", columns="correction", values=["true_auc", "cv_auc"])
        best_true = dm["true_auc"][CORR].idxmax(axis=1)
        best_cv = dm["cv_auc"][CORR].idxmax(axis=1)
        rows.append({"strength": strength, "model": m, "cv_picks_true_best_pct": 100 * (best_true == best_cv).mean(),
                     "cv_picks_none_pct": 100 * (best_cv == "none").mean(),
                     "truth_best_none_pct": 100 * (best_true == "none").mean()})
sel = pd.DataFrame(rows)
sel.to_csv(RESULTS / "E03_selection.csv", index=False)
print("\nModel selection by random CV:\n", sel.round(1).to_string(index=False))

plt.rcParams.update({"figure.dpi": 150, "font.size": 8})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
g = df[df.correction == "none"].groupby(["strength", "model"])[["true_auc", "cv_auc", "leakage"]].mean().reset_index()
for m in ORDER:
    x = g[g.model == m]
    ax[0].plot(x.strength, x.true_auc, "-o", c=COL[m], label=m, ms=3)
    ax[0].plot(x.strength, x.cv_auc, "--", c=COL[m], lw=0.8)
    ax[1].plot(x.strength, x.leakage, "-o", c=COL[m], ms=3)
ax[0].set_title("uncorrected models\ntrue AUC (solid) vs random-CV AUC (dashed)")
ax[0].set_xlabel("bias strength")
ax[0].legend(fontsize=6)
ax[1].set_title("uncorrected models\nbias leakage (partial R² of accessibility)")
ax[1].set_xlabel("bias strength")
c2 = df[df.strength == 2].groupby(["correction", "model"]).leakage.mean().unstack()[ORDER].loc[CORR]
xx = np.arange(len(CORR))
for i, m in enumerate(ORDER):
    ax[2].bar(xx + (i - 2) * 0.16, c2[m], 0.16, color=COL[m], label=m)
ax[2].set_xticks(xx, CORR)
ax[2].set_title("strong bias (strength 2)\nleakage by correction")
fig.tight_layout()
fig.savefig(FIGURES / "E03_capacity_leakage.png")
