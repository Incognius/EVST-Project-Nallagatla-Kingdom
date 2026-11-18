"""Figure for E23: how many survey sites does it take to choose the bias-corrected model?"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evst.paths import FIGURES, RESULTS

BLUE, ORANGE, GRAY, INK, MUTED, RED = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e", "#c03a2b"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
systems = [("kerala", "Kerala Bird Atlas (birds)", "Correction (BRT)", "atlas cells"),
           ("glc", "GLC24 Europe (plants)", "Correction (MLP)", "survey plots")]
systems = [s for s in systems if (RESULTS / "E23" / f"E23_{s[0]}_summary.csv").exists()]
fig, ax = plt.subplots(2, len(systems), figsize=(5.4 * len(systems), 7.2), squeeze=False)

for k, (sysname, title, contrast, unit) in enumerate(systems):
    S = pd.read_csv(RESULTS / "E23" / f"E23_{sysname}_summary.csv")
    CV = pd.read_csv(RESULTS / "E23" / f"E23_{sysname}_cv_summary.csv")
    S = S[S.contrast == contrast]
    full = S[S.n == "all"].n_sites.iloc[0]
    for design, col, ls in (("random", BLUE, "-"), ("clustered", ORANGE, "--")):
        d = S[S.design == design]
        x = d.n_sites.to_numpy()
        lab = "sites placed at random" if design == "random" else "sites in spatial clusters"
        ax[0, k].plot(x, d.p_sig_correct * 100, ls, color=col, lw=2, marker="o", ms=4, label=lab)
        ax[0, k].plot(x, d.p_sig_wrong * 100, ls, color=col, lw=1, alpha=0.5)
        ax[1, k].plot(x, d.species_agree * 100, ls, color=col, lw=2, marker="o", ms=4, label=lab)
    # survey-free checks on the same reference halves
    c = CV[CV.contrast == contrast].set_index("check")
    ax[0, k].axhline(0, color=RED, lw=2)
    ax[0, k].text(x.min(), 11, "random CV on all citizen-science records: picks the uncorrected model (wrong)",
                  color=RED, fontsize=7.5, va="bottom")
    for chk, col in (("random CV", RED), ("target-group CV", MUTED)):
        if chk in c.index:
            v = c.loc[chk, "species_agree"] * 100
            ax[1, k].axhline(v, color=col, lw=1.5, ls=":")
            ax[1, k].text(full, v + 1.2, f"{chk}: {v:.0f}%", color=col, fontsize=7.5, ha="right")
    ax[1, k].axhline(50, color=GRAY, lw=0.8)
    for r in (0, 1):
        ax[r, k].set_xscale("log")
        ticks = [t for t in (10, 25, 50, 100, 250, 500, 1000, 5000, 20000) if x.min() <= t <= full * 1.05]
        ax[r, k].set_xticks(ticks, [f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks])
        ax[r, k].minorticks_off()
        ax[r, k].set_xlabel(f"Independent {unit} in the survey (of {int(full):,} in the pool)")
    ax[0, k].set_ylim(-2, 104)
    ax[1, k].set_ylim(0, 100)
    ax[0, k].set_ylabel("% of surveys that confidently pick\nthe right model (thin: confidently wrong)")
    ax[1, k].set_ylabel("% of species where the survey\nmakes the right per-species choice")
    ax[0, k].set_title(f"{'AB'[k]}  {title}: one choice for all species", loc="left", fontsize=9.5, color=INK)
    ax[1, k].set_title(f"{'CD'[k]}  {title}: species-by-species choice", loc="left", fontsize=9.5, color=INK)
    ax[0, k].legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(FIGURES / "E23_survey_budget.png", dpi=180)
print("saved", FIGURES / "E23_survey_budget.png")
