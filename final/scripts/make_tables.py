import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in (Path(__file__).resolve().parents[2] / "biodiv-gaps", Path(__file__).resolve().parents[2]) if (p / "src" / "evst").exists())
sys.path.insert(0, str(ROOT / "src"))
from evst.data import systems
from evst.paths import PROCESSED

R = ROOT / "results"
FIN = Path(__file__).resolve().parents[1]
TAB = FIN / "paper" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
BB = pd.read_csv(FIN / "results" / "block_bootstrap_cis.csv")
macros = {}


def f3(v):
    return f"{v:.3f}"


def sg(v, d=3):
    return f"{v:+.{d}f}".replace("-", "$-$")


def pval(p):
    if p < 1e-4:
        e = int(np.floor(np.log10(p)))
        return f"$<10^{{{e + 1}}}$"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"


def ci(analysis, contrast):
    r = BB[(BB.analysis == analysis) & (BB.contrast == contrast)]
    if not len(r):
        return ""
    r = r.iloc[0]
    return f"[{sg(r.ci_lo)}, {sg(r.ci_hi)}]"


def write(name, body):
    (TAB / f"{name}.tex").write_text(body, encoding="utf-8")
    print("table", name)


def hit_rates(S, sc):
    Y, mask = S["Y"], S["R"] == 0
    out = {}
    for k, v in sc.items():
        rk = np.full(v.shape, np.nan)
        for j in range(v.shape[1]):
            mm = mask[:, j]
            if mm.sum():
                rk[mm, j] = v[mm, j].argsort().argsort() / max(mm.sum() - 1, 1)
        top = (rk >= 0.9) & mask
        out[k] = Y[top].mean()
    out["base"] = Y[mask].mean()
    return out


K = systems.get("kba")
T = systems.get("taiwan")
hk = hit_rates(K, dict(np.load(PROCESSED / "E16_brt_scores.npz")))
ht = hit_rates(T, dict(np.load(PROCESSED / "E17_scores_taiwan.npz")))
k9 = pd.read_csv(R / "E09b_kba_unrecorded_species_auc.csv")
t17 = pd.read_csv(R / "E17_missing_pair_auc_taiwan.csv")
kn = pd.read_csv(R / "E11_neural_species_auc.csv").pivot_table(index="species", columns="loss", values="auc_missing")
macros.update(KpairsN=f"{int((K['R'] == 0).sum()):,}", TpairsN=f"{int((T['R'] == 0).sum()):,}",
              Kbase=f"{100 * hk['base']:.1f}", Tbase=f"{100 * ht['base']:.1f}",
              KhitTGB=f"{100 * hk['BRT_tgb']:.1f}", ThitTGB=f"{100 * ht['BRT_tgb']:.1f}",
              KhitNaive=f"{100 * hk['BRT_naive']:.1f}", ThitNaive=f"{100 * ht['BRT_naive']:.1f}",
              KhitNear=f"{100 * hk['nearest_record']:.1f}", ThitNear=f"{100 * ht['nearest_record']:.1f}",
              KnSp=str(len(k9)), TnSp=str(len(t17)), Kcells=str(len(K["ev"])), Tsites=str(len(T["ev"])),
              Kcand=str(len(K["cand"])), Tcand=str(len(T["cand"])))
rows = [("Nearest existing record", k9.nearest_record.mean(), t17.nearest_record.mean(), hk["nearest_record"], ht["nearest_record"]),
        ("GLM, target-group background", k9.GLM_tgb.mean(), None, None, None),
        ("BRT, random background", k9.BRT_naive.mean(), t17.BRT_naive.mean(), hk["BRT_naive"], ht["BRT_naive"]),
        ("Neural, random negatives", kn["an_ssdl"].mean(), t17.NN_an_ssdl.mean(), None, None),
        ("Neural, effort-aware likelihood", kn["occ_effort"].mean(), t17.NN_occ_effort.mean(), None, None),
        ("BRT, target-group background", k9.BRT_tgb.mean(), t17.BRT_tgb.mean(), hk["BRT_tgb"], ht["BRT_tgb"])]
body = []
for nm, a, b, c, d in rows:
    cells = [nm, f3(a), f3(b) if b is not None else "--", f"{100 * c:.1f}" if c is not None else "--",
             f"{100 * d:.1f}" if d is not None else "--"]
    if nm.startswith("BRT, target"):
        cells = [f"\\textbf{{{x}}}" for x in cells]
    body.append(" & ".join(cells) + r" \\")
body.append(r"\midrule")
body.append(f"Chance (AUC) and survey prevalence (hit rate) & 0.500 & 0.500 & {100 * hk['base']:.1f} & {100 * ht['base']:.1f} \\\\")
body.append(r"\midrule")
for lab, kc, tc, an_k, an_t in (("BRT: target-group $-$ random background", k9.BRT_tgb - k9.BRT_naive, t17.BRT_tgb - t17.BRT_naive,
                                 "kerala_missing_pairs", "taiwan_missing_pairs"),
                                ("BRT target-group $-$ nearest record", k9.BRT_tgb - k9.nearest_record, t17.BRT_tgb - t17.nearest_record,
                                 "kerala_missing_pairs", "taiwan_missing_pairs")):
    con = "BRT_tgb - BRT_naive" if "random" in lab else "BRT_tgb - nearest_record"
    body.append(f"{lab} & {sg(kc.mean())} & {sg(tc.mean())} & \\multicolumn{{2}}{{l}}{{\\small K {ci(an_k, con)}; T {ci(an_t, con)}}} \\\\")
    body.append(f"\\quad species improved & {100 * (kc > 0).mean():.0f}\\% & {100 * (tc > 0).mean():.0f}\\% & & \\\\")
write("missing_pairs", "\n".join(body))
macros.update(KgainTGB=f3(( k9.BRT_tgb - k9.BRT_naive).mean()), TgainTGB=f3((t17.BRT_tgb - t17.BRT_naive).mean()),
              KgainNear=f3((k9.BRT_tgb - k9.nearest_record).mean()), TgainNear=f3((t17.BRT_tgb - t17.nearest_record).mean()),
              KaucTGB=f3(k9.BRT_tgb.mean()), TaucTGB=f3(t17.BRT_tgb.mean()),
              KpctTGB=f"{100 * ((k9.BRT_tgb - k9.BRT_naive) > 0).mean():.0f}",
              TpctTGB=f"{100 * ((t17.BRT_tgb - t17.BRT_naive) > 0).mean():.0f}")
kci = BB[(BB.analysis == "kerala_missing_pairs") & (BB.contrast == "BRT_tgb - BRT_naive")].iloc[0]
tci = BB[(BB.analysis == "taiwan_missing_pairs") & (BB.contrast == "BRT_tgb - BRT_naive")].iloc[0]
nci = BB[(BB.analysis == "taiwan_missing_pairs") & (BB.contrast == "NN_occ_effort - BRT_tgb")].iloc[0]
macros.update(KgainCI=f"[{kci.ci_lo:.3f}, {kci.ci_hi:.3f}]", TgainCI=f"[{tci.ci_lo:.3f}, {tci.ci_hi:.3f}]",
              TnnCI=f"[{sg(nci.ci_lo)}, {sg(nci.ci_hi)}]", TnnDelta=sg(nci.delta), Kblocks=str(int(kci.n_blocks)),
              Tblocks=str(int(tci.n_blocks)))

e2 = pd.read_csv(R / "E02_reversal.csv").set_index("regime")
kc = pd.read_csv(R / "E09b_kba_cv_vs_truth.csv")
tc = pd.read_csv(R / "E17_cv_vs_truth_taiwan.csv")
ti = pd.read_csv(R / "E17_cv_inversion_taiwan.csv").iloc[0]
from scipy.stats import spearmanr
kcv, ktr = kc.cv_tgb - kc.cv_naive, kc.kba_tgb - kc.kba_naive
tcv, ttr = tc.cv_tgb - tc.cv_naive, tc.truth_tgb - tc.truth_naive
body = [f"disdat (6 regions, 4 algorithms) & 896 fits & {e2.loc['random CV', 'says_tgb_better_pct']:.1f}\\% & 62.0\\% & "
        f"{e2.loc['random CV', 'agrees_with_truth_pct']:.0f}\\% & {sg(e2.loc['random CV', 'spearman_delta_vs_truth'], 2)} \\\\",
        f"disdat, spatial-block CV & 896 fits & {e2.loc['spatial-block CV', 'says_tgb_better_pct']:.1f}\\% & 62.0\\% & "
        f"{e2.loc['spatial-block CV', 'agrees_with_truth_pct']:.0f}\\% & {sg(e2.loc['spatial-block CV', 'spearman_delta_vs_truth'], 2)} \\\\",
        f"disdat, spatial CV vs target-group sites & 896 fits & {e2.loc['spatial CV vs target-group sites', 'says_tgb_better_pct']:.1f}\\% & 62.0\\% & "
        f"{e2.loc['spatial CV vs target-group sites', 'agrees_with_truth_pct']:.0f}\\% & {sg(e2.loc['spatial CV vs target-group sites', 'spearman_delta_vs_truth'], 2)} \\\\",
        r"\midrule",
        f"Kerala (BRT, random CV) & {len(kc)} species & {100 * (kcv > 0).mean():.1f}\\% & {100 * (ktr > 0).mean():.1f}\\% & "
        f"{100 * ((kcv > 0) == (ktr > 0)).mean():.0f}\\% & {sg(spearmanr(kcv, ktr)[0], 2)} \\\\",
        f"Taiwan (BRT, random CV) & {len(tc)} species & {100 * (tcv > 0).mean():.1f}\\% & {100 * (ttr > 0).mean():.1f}\\% & "
        f"{100 * ((tcv > 0) == (ttr > 0)).mean():.0f}\\% & {sg(spearmanr(tcv, ttr)[0], 2)} \\\\"]
write("cv_inversion", "\n".join(body))
macros.update(KcvPick=f"{100 * (kcv > 0).mean():.1f}", KtrPick=f"{100 * (ktr > 0).mean():.1f}",
              TcvPick=f"{100 * (tcv > 0).mean():.1f}", TtrPick=f"{100 * (ttr > 0).mean():.1f}",
              KoptNaive=f3((kc.cv_naive - kc.kba_naive).mean()), KoptTGB=sg((kc.cv_tgb - kc.kba_tgb).mean()),
              ToptNaive=f3(ti.optimism_naive), ToptTGB=sg(ti.optimism_tgb))

body = []
for name, f, lab, key in (("Kerala, train $\\le$2014", "E15_truth_source_summary_train2014.csv", "kba", "flip_kba_2014"),
                          ("Kerala, train $\\le$2017", "E15_truth_source_summary_train2017.csv", "kba", None),
                          ("Taiwan, train $\\le$2016", "E15_truth_source_summary_taiwan_train2016.csv", "twbbs", "flip_taiwan_2016"),
                          ("Taiwan, train $\\le$2018", "E15_truth_source_summary_taiwan_train2018.csv", "twbbs", "flip_taiwan_2018")):
    S = pd.read_csv(R / f)
    for c, short in (("BRT_tgb - BRT_naive", "BRT"), ("NN_occ_effort - NN_an_ssdl", "Neural")):
        a = S[(S.truth == f"{lab}_same_cells") & (S.contrast == c)].iloc[0]
        b = S[(S.truth == "inat_later_same_cells") & (S.contrast == c)].iloc[0]
        cia = ci(f"{key}_survey", c) if key else "--"
        cib = ci(f"{key}_later_inat", c) if key else "--"
        body.append(f"{name} & {short} & {int(a.n_species)} & {sg(a.delta)} & {cia} & {100 * a.pct_corrected_better:.0f}\\% & "
                    f"{sg(b.delta)} & {cib} & {100 * b.pct_corrected_better:.0f}\\% \\\\")
    body.append(r"\addlinespace")
write("flip", "\n".join(body[:-1]))

L = pd.read_csv(R / "E11_neural_summary.csv")
order = [("an_ssdl", "SINR assume-negative, random locations", "random land"),
         ("maxent_random", "DeepMaxent-style Poisson, random background", "random land"),
         ("an_full", "SINR default (random + other species)", "mixed"),
         ("an_slds", "SINR, same-location other species", "sampled pixels"),
         ("maxent_tgb", "DeepMaxent-style Poisson, target-group background", "sampled pixels"),
         ("occ_effort", "Occupancy $\\times$ detection, $1-(1-p)^{U}$ (ours)", "sampled pixels, effort-weighted")]
body = []
for k, nm, neg in order:
    r = L[(L.model == k) & (L.ref == "BRT_tgb")]
    m = kn[k].mean()
    d = r.iloc[0] if len(r) else None
    body.append(f"{nm} & {neg} & {f3(m)} & {sg(d.delta) if d is not None else '--'} \\\\")
body.append(f"BRT, target-group background & sampled pixels & {f3(k9.BRT_tgb.mean())} & 0 \\\\")
write("neural_losses", "\n".join(body))

sb = pd.read_csv(R / "E01_sampbias.csv")
body = []
for _, r in sb.iterrows():
    nm = r.source.replace("group:", "").replace("_", " and ")
    body.append(f"{nm} & {int(r.records):,} & " + " & ".join(f"{r[c]:.2f}" for c in
               ("w_d_road_all", "w_d_road_major", "w_d_town", "w_d_city", "w_d_river", "w_d_reserve")) + r" \\")
write("sampbias", "\n".join(body))

E5 = pd.read_csv(R / "E05_full_scores_blockboot.csv")
body = []
for g in ("all", "odonates", "birds", "butterflies_moths", "reptiles", "amphibians", "mammals", "plants"):
    r = E5[(E5.group == g) & (E5.origin == 2018) & (E5.model == "bayes") & (E5.stratum == "all")]
    if len(r):
        r = r.iloc[0]
        body.append(f"{g.replace('_', ' and ')} & {int(r.n_cells)} & {sg(r.delta_vs_effort_glm)} & [{sg(r.ci_lo)}, {sg(r.ci_hi)}] \\\\")
write("discovery_taxa", "\n".join(body))

nm = FIN / "paper" / "numbers.tex"
nm.write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n", encoding="utf-8")
print(macros)
