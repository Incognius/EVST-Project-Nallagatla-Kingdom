import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evst.paths import FIGURES, RESULTS

plt.rcParams.update({"figure.dpi": 150, "font.size": 8.5})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

D = pd.read_csv(RESULTS / "E11_neural_species_auc.csv")
B = pd.read_csv(RESULTS / "E09b_kba_unrecorded_species_auc.csv").set_index("species")
W = D.pivot_table(index="species", columns="loss", values="auc_missing").join(B[["BRT_tgb", "BRT_naive", "nearest_record"]], how="inner").dropna()
order = ["nearest_record", "an_ssdl", "maxent_random", "an_full", "BRT_naive", "an_slds", "maxent_tgb", "occ_effort", "BRT_tgb"]
tgb = {"an_slds", "maxent_tgb", "occ_effort", "BRT_tgb"}
m = W[order].mean()
ax[0].barh(range(len(order)), m.values, color=["#D2601A" if k in tgb else "#9C9A90" for k in order])
ax[0].set_yticks(range(len(order)), order)
ax[0].set_xlim(0.6, 0.78)
ax[0].set_xlabel("mean AUC on iNat-unrecorded pairs (atlas truth)")
ax[0].set_title("a  Loss family: target-group > random negatives", loc="left")

E = pd.read_csv(RESULTS / "E13_atlas_oracle_allocation.csv")
g = E.groupby(["model", "strategy", "n_cells"]).auc_all.mean()
for (mod, s), col in {("po+pa", "spread"): "#4F6B35", ("po+pa", "random"): "#D2601A", ("po+pa", "low_effort"): "#C9A227",
                      ("po+pa", "uncertainty"): "#7E7C72", ("po+pa", "discovery"): "#8E3510", ("pa_only", "random"): "#3B6EA5"}.items():
    q = g.loc[mod, s]
    q = q[q.index > 0] if mod == "pa_only" else q
    ax[1].plot(np.maximum(q.index, 5), q.values, marker="o", ms=3, color=col, ls="--" if mod == "pa_only" else "-",
               label=f"{s}" + (" (atlas only, no iNat)" if mod == "pa_only" else ""))
ax[1].set_xscale("log")
ax[1].set_xlabel("atlas cells revealed (0 plotted at 5)")
ax[1].set_ylabel("mean AUC on held-out atlas blocks")
ax[1].legend(fontsize=6.5)
ax[1].set_title("b  Where to survey next", loc="left")

S = pd.concat([pd.read_csv(RESULTS / f"E15_truth_source_summary_train{y}.csv").assign(train=y) for y in (2014, 2017)])
S = S[S.truth != "kba_all_unrecorded"]
lab = {"kba_same_cells": "atlas truth", "inat_later_same_cells": "later iNat truth"}
x = 0
ticks = []
for (tr, c), q in S.groupby(["train", "contrast"]):
    for t, col in (("kba_same_cells", "#4F6B35"), ("inat_later_same_cells", "#8E3510")):
        v = q[q.truth == t].delta.values[0]
        ax[2].bar(x, v, color=col, label=lab[t] if x < 2 else None)
        x += 1
    ticks.append((x - 1.5, f"{'BRT' if 'BRT' in c else 'NN'}\n<= {tr}"))
    x += 0.6
ax[2].axhline(0, c="k", lw=0.6)
ax[2].set_xticks([t[0] for t in ticks], [t[1] for t in ticks])
ax[2].set_ylabel("AUC gain of bias-corrected model")
ax[2].legend(fontsize=7)
ax[2].set_title("c  Same pairs, opposite verdicts", loc="left")
fig.tight_layout()
fig.savefig(FIGURES / "E11_E15_phase_c.png")
