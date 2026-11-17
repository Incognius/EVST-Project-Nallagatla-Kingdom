import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from evst.paths import FIGURES, RESULTS

BLUE, ORANGE, INK, MUTED, GRID = "#3B6EA5", "#D2601A", "#2B2B2B", "#6F6D66", "#E4E2DC"
plt.rcParams.update({"figure.dpi": 150, "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(13, 8.4))

runs = [("Kerala ≤2014", "E15_truth_source_summary_train2014.csv", "kba"),
        ("Taiwan ≤2016", "E15_truth_source_summary_taiwan_train2016.csv", "twbbs"),
        ("Taiwan ≤2018", "E15_truth_source_summary_taiwan_train2018.csv", "twbbs")]
x, ticks = 0.0, []
for name, f, lab in runs:
    S = pd.read_csv(RESULTS / f)
    for c, short in (("BRT_tgb - BRT_naive", "BRT"), ("NN_occ_effort - NN_an_ssdl", "NN")):
        for t, col, lb in ((f"{lab}_same_cells", BLUE, "survey truth"), ("inat_later_same_cells", ORANGE, "later iNaturalist truth")):
            v = S[(S.truth == t) & (S.contrast == c)].delta.values[0]
            ax[0, 0].bar(x, v, width=0.8, color=col, label=lb if x < 2 else None)
            x += 1
        ticks.append((x - 1.5, short + "\n" + name.replace(" ", "\n")))
        x += 0.7
ax[0, 0].axhline(0, c=MUTED, lw=0.8)
ax[0, 0].set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7.5)
ax[0, 0].set_ylabel("AUC gain of bias-corrected model")
ax[0, 0].legend(frameon=False, fontsize=7.5)
ax[0, 0].set_title("a  Same pairs, opposite verdicts — replicated in Taiwan", loc="left", color=INK)

K = pd.read_csv(RESULTS / "E16_rescored_contrasts.csv")
T = pd.read_csv(RESULTS / "E17b_taiwan_occupancy_rescored.csv")
items = [(f"Kerala · {v}", K[(K.variant == k) & (K.contrast == "BRT_tgb - BRT_naive")].delta.values[0])
         for k, v in (("atlas", "atlas as given"), ("drop_pfa_gt_0.1", "likely false absences dropped"),
                      ("soft_occupancy", "soft occupancy labels"), ("wet_only", "wet season only"), ("dry_only", "dry season only"))]
items += [(f"Taiwan · {v}", T[(T.variant == k) & (T.contrast == "BRT_tgb - BRT_naive")].delta.values[0])
          for k, v in (("survey", "survey as given"), ("drop_pfa_gt_0.1", "likely false absences dropped"))]
yy = np.arange(len(items))[::-1]
ax[0, 1].barh(yy, [v for _, v in items], height=0.6, color=BLUE)
for y, (_, v) in zip(yy, items):
    ax[0, 1].text(v + 0.001, y, f"+{v:.3f}", va="center", fontsize=7.5, color=INK)
ax[0, 1].set_yticks(yy, [n for n, _ in items], fontsize=7.5)
ax[0, 1].set_xlim(0, 0.055)
ax[0, 1].set_xlabel("AUC gain, target-group over random background (iNat-unrecorded pairs)")
ax[0, 1].set_title("b  Gain survives every occupancy-corrected truth", loc="left", color=INK)

order = ["mf", "knn", "nearest_record", "sar_pu", "env_tg", "env_mf_tg", "expo_em", "occ_unit", "BRT_tgb"]
lab = {"mf": "logistic MF", "knn": "assemblage kNN", "nearest_record": "nearest record", "sar_pu": "SAR-PU (IPS)",
       "env_tg": "env, target-group", "env_mf_tg": "env + latent factors", "expo_em": "exposure-EM (MNAR)",
       "occ_unit": "env + effort likelihood", "BRT_tgb": "BRT target-group"}
for sys_, f, mk, fc in (("Kerala", "E18_species_auc.csv", "o", BLUE), ("Taiwan", "E18_species_auc_taiwan.csv", "s", "white")):
    D = pd.read_csv(RESULTS / f)
    m = D[D.stratum == "all_missing"][order].mean()
    ax[1, 0].scatter(m.values, np.arange(len(order)), marker=mk, s=42, facecolor=fc, edgecolor=BLUE, linewidth=1.4,
                     label=sys_, zorder=3)
ax[1, 0].set_yticks(np.arange(len(order)), [lab[o] for o in order], fontsize=7.5)
ax[1, 0].axvline(0.5, c=MUTED, lw=0.8, ls=":")
ax[1, 0].grid(axis="x", color=GRID, lw=0.6)
ax[1, 0].set_xlabel("mean AUC on iNat-unrecorded pairs (survey truth)")
ax[1, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
ax[1, 0].set_title("c  Co-occurrence and MNAR debiasing add nothing", loc="left", color=INK)

P = pd.read_csv(RESULTS / "E19_paired_vs_random.csv")
P = P[P.n == 160]
strat = ["uncertainty", "discovery", "effort_disc_unc", "coverage_unc", "adaptive", "thompson", "low_effort", "spread"]
slab = {"uncertainty": "uncertainty", "discovery": "expected discoveries", "effort_disc_unc": "effort-discounted uncertainty",
        "coverage_unc": "coverage + uncertainty", "adaptive": "adaptive λ", "thompson": "Thompson batch",
        "low_effort": "fewest visits", "spread": "spatial spread"}
systems = ["kba", "taiwan", "disdat_CAN", "disdat_NZ", "disdat_SWI"]
M = P.pivot_table(index="strategy", columns="system", values="delta_vs_random").loc[strat, systems]
B = P.pivot_table(index="strategy", columns="system", values="splits_better").loc[strat, systems]
cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#F2F1EE", BLUE])
lim = 0.02
im = ax[1, 1].imshow(M.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax[1, 1].text(j, i, f"{M.values[i, j]:+.3f}\n{int(B.values[i, j])}/10", ha="center", va="center", fontsize=6.8,
                      color=INK)
ax[1, 1].set_xticks(range(len(systems)), ["Kerala", "Taiwan", "Canada", "New Zealand", "Switzerland"], fontsize=7.5)
ax[1, 1].set_yticks(range(len(strat)), [slab[s] for s in strat], fontsize=7.5)
ax[1, 1].spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax[1, 1], fraction=0.035, pad=0.02)
cb.set_label("AUC vs random placement at 160 sites", fontsize=7.5)
ax[1, 1].set_title("d  Where to survey: uncertainty loses everywhere", loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIGURES / "E16_E19_phase_d.png")
