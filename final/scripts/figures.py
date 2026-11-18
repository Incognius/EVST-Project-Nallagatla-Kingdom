import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from pyproj import Transformer

ROOT = next(p for p in (Path(__file__).resolve().parents[2] / "biodiv-gaps", Path(__file__).resolve().parents[2]) if (p / "src" / "evst").exists())
sys.path.insert(0, str(ROOT / "src"))
from evst.data import records, systems
from evst.data.rasters import load_stack
from evst.paths import PROCESSED

R = ROOT / "results"
FIN = Path(__file__).resolve().parents[1]
SLIDES = bool(os.environ.get("SLIDES"))
OUT = FIN / "slides" / "figs" if SLIDES else FIN / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BG = "#F7F6EF" if SLIDES else "white"
BLUE, ORANGE, INK, MUTED, GRID, PALE = "#3B6EA5", "#D2601A", "#2B2B2B", "#6F6D66", "#E4E2DC", "#F2F1EE"
DARKBLUE, DARKOR = "#1F3F66", "#8E3510"
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 8, "font.family": "DejaVu Sans",
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 8.5, "axes.titleweight": "bold",
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
ORAMP = LinearSegmentedColormap.from_list("or", ["#FBEBDD", "#F0A25C", ORANGE, DARKOR])
BRAMP = LinearSegmentedColormap.from_list("bl", ["#E3ECF6", "#8FB0D6", BLUE, DARKBLUE])
DIV = LinearSegmentedColormap.from_list("div", [ORANGE, PALE, BLUE])


def title(ax, letter, text):
    ax.set_title(f"{letter}  {text}", loc="left", color=INK)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", name)


def mapax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


def fig_study_systems():
    layers, tr, mask = load_stack()
    H, W = mask.shape
    rec = records.load()
    f = 5
    Hc, Wc = H // f, W // f
    cnt = np.zeros((Hc, Wc))
    rr, cc = rec.row.to_numpy() // f, rec.col.to_numpy() // f
    ok = (rr < Hc) & (cc < Wc)
    np.add.at(cnt, (rr[ok], cc[ok]), 1)
    land = mask[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).any((1, 3))
    img = np.where(land, cnt, np.nan)
    ext = [tr.c / 1e3, (tr.c + Wc * f * tr.a) / 1e3, (tr.f + Hc * f * tr.e) / 1e3, tr.f / 1e3]
    K = pd.read_csv(R / "E09_kba_cells.csv")
    ll = K.kcell.str.strip("[]").str.split(",", expand=True).astype(float)
    kx, ky = Transformer.from_crs(4326, "EPSG:32643", always_xy=True).transform(ll[0].values, ll[1].values)
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.15], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    bgimg = np.where(land, 0.0, np.nan)
    ax.imshow(bgimg, extent=ext, cmap=LinearSegmentedColormap.from_list("g", [GRID, GRID]), interpolation="nearest")
    im = ax.imshow(np.where(img > 0, img, np.nan), extent=ext, cmap=ORAMP, norm=LogNorm(1, img[np.isfinite(img)].max()),
                   interpolation="nearest")
    ax.scatter(kx / 1e3, ky / 1e3, s=0.6, c=DARKBLUE, lw=0, alpha=0.6)
    mapax(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, shrink=0.55, location="left")
    cb.set_label("iNaturalist records per 5 km cell", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.text(0.98, 0.98, "493,227 records\n10,219 species\n(iNaturalist, to 2026)\n\nblue dots: Kerala\nBird Atlas cells",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK)
    title(ax, "a", "Western Ghats, all taxa")
    ax = fig.add_subplot(gs[1])
    ax.scatter(kx / 1e3, ky / 1e3, s=11, marker="s", c=K.frac_missed, cmap=BRAMP, vmin=0, vmax=1, lw=0)
    vis = K.inat_visits.to_numpy() == 0
    ax.scatter(kx[vis] / 1e3, ky[vis] / 1e3, s=1.2, c="white", lw=0)
    mapax(ax)
    sm = plt.cm.ScalarMappable(cmap=BRAMP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.01, shrink=0.55, orientation="horizontal", location="bottom")
    cb.set_label("share of atlas species iNaturalist never\nrecorded in the cell (2015-2020 atlas)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.text(0.02, 0.02, f"844 cells, 6.6 km\n{vis.sum()} with no iNat\nbird visit to 2020\n(white dot)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=INK)
    title(ax, "b", "Kerala Bird Atlas")
    T = systems.get("taiwan")
    tl, ttr, tmask = __import__("evst.data.taiwan", fromlist=["load_stack"]).load_stack()
    th, tw = tmask.shape
    b = T["birds"]
    tc = np.bincount(b.li.to_numpy(), minlength=len(T["land"])).astype(float)
    timg = np.full(th * tw, np.nan)
    timg[T["land"]] = tc
    timg = timg.reshape(th, tw)
    text = [ttr.c / 1e3, (ttr.c + tw * ttr.a) / 1e3, (ttr.f + th * ttr.e) / 1e3, ttr.f / 1e3]
    ax = fig.add_subplot(gs[2])
    ax.imshow(np.where(tmask, 0.0, np.nan), extent=text, cmap=LinearSegmentedColormap.from_list("g", [GRID, GRID]),
              interpolation="nearest")
    ax.imshow(np.where(timg > 0, timg, np.nan), extent=text, cmap=ORAMP, norm=LogNorm(1, np.nanmax(timg)),
              interpolation="nearest")
    sx, sy = Transformer.from_crs(4326, "EPSG:32651", always_xy=True).transform(T["cxy"][:, 0], T["cxy"][:, 1])
    Y, Rr = T["Y"], T["R"]
    fm = ((Y == 1) & (Rr == 0)).sum(1) / np.maximum(Y.sum(1), 1)
    ax.scatter(sx / 1e3, sy / 1e3, s=9, c=fm, cmap=BRAMP, vmin=0, vmax=1, lw=0.3, edgecolor="white")
    mapax(ax)
    ax.text(0.0, 1.0, f"orange: {len(b):,} iNat records\nblue: {len(T['ev'])} survey sites",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK)
    title(ax, "c", "Taiwan Breeding Bird Survey")
    save(fig, "F1_study_systems")


def fig_bias_anatomy():
    ri = pd.read_csv(R / "E01_road_intensity.csv")
    ol = pd.read_csv(R / "E01_oliver.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    a = ri[ri.group == "all"]
    x = np.arange(len(a))
    ax[0].bar(x, a.intensity_ratio, color=[ORANGE if v > 1 else BLUE for v in a.intensity_ratio], width=0.7)
    for xi, v in zip(x, a.intensity_ratio):
        ax[0].text(xi, v + 0.05, f"{v:.2f}x", ha="center", va="bottom", fontsize=6.5 if v > 0.5 else 5.8, color=INK,
                   rotation=0 if v > 0.5 else 90)
    ax[0].axhline(1, color=MUTED, lw=0.7, ls=":")
    ax[0].set_xticks(x, [b.replace("-", "–") for b in a.band_km], fontsize=6.3, rotation=40, ha="right")
    ax[0].set_xlabel("distance to nearest road (km)")
    ax[0].set_ylabel("share of records / share of land")
    title(ax[0], "a", "Records cling to roads")
    o = ol[(ol.source == "all") & ol.year.between(2008, 2025)]
    ax[1].bar(o.year, o.records / 1000, color=BLUE, width=0.75)
    ax[1].set_ylabel("records per year (thousands)")
    title(ax[1], "b", "Volume: 61x since 2008")
    ax[2].plot(o.year, o.effectiveness_per_1000, color=ORANGE, lw=2, marker="o", ms=3)
    ax[2].set_ylabel("new species x 10 km cells\nper 1,000 records")
    ax[2].set_ylim(0, None)
    ax[2].grid(axis="y", color=GRID, lw=0.6)
    title(ax[2], "c", "Novelty per record: -56%")
    for a_ in ax[1:]:
        a_.set_xticks([2010, 2015, 2020, 2025])
    fig.tight_layout()
    save(fig, "F2_bias_anatomy")


def fig_virtual():
    d = pd.read_csv(R / "E03_summary.csv")
    n = d[d.correction == "none"].groupby("strength")[["true_auc", "cv_auc"]].mean()
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    ax[0].plot(n.index, n.true_auc, color=BLUE, lw=2, marker="o", ms=4, label="true AUC (known truth)")
    ax[0].plot(n.index, n.cv_auc, color=ORANGE, lw=2, marker="s", ms=4, label="random-CV AUC")
    for s in n.index:
        up = n.true_auc[s] > n.cv_auc[s]
        ax[0].text(s, n.true_auc[s] + (0.012 if up else -0.024), f"{n.true_auc[s]:.2f}", ha="center", fontsize=6.5, color=BLUE)
        ax[0].text(s, n.cv_auc[s] + (-0.024 if up else 0.012), f"{n.cv_auc[s]:.2f}", ha="center", fontsize=6.5, color=ORANGE)
    ax[0].set_xticks([0, 1, 2], ["none", "realistic", "exaggerated"])
    ax[0].set_xlabel("observer-bias strength")
    ax[0].set_ylabel("mean AUC, 5 uncorrected models")
    ax[0].set_ylim(0.65, 0.99)
    ax[0].legend(loc="upper left", fontsize=6.3)
    title(ax[0], "a", "Truth falls, CV rises")
    base = d[(d.correction == "none") & (d.strength == 0)].set_index("model").leakage
    s2 = d[(d.correction == "none") & (d.strength == 2)].set_index("model").leakage
    ex = (s2 - base).sort_values()
    ax[1].barh(np.arange(len(ex)), ex.values, color=[BLUE if m == "GLM" else ORANGE for m in ex.index], height=0.6)
    for i, v in enumerate(ex.values):
        ax[1].text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=6.5, color=INK)
    ax[1].set_yticks(np.arange(len(ex)), ex.index)
    ax[1].set_xlabel("excess bias leakage (partial R$^2$)")
    title(ax[1], "b", "Flexible models leak more")
    c = d[d.strength == 2].groupby("correction").true_auc.mean().reindex(["none", "thin", "biascov", "tgb"])
    lab = ["none", "thinning", "bias\ncovariate", "target-\ngroup bg"]
    ax[2].bar(np.arange(4), c.values, color=[MUTED, MUTED, MUTED, BLUE], width=0.65)
    for i, v in enumerate(c.values):
        ax[2].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=6.5, color=INK)
    ax[2].set_xticks(np.arange(4), lab, fontsize=6.3)
    ax[2].set_ylim(0.68, 0.77)
    ax[2].set_ylabel("true AUC at strength 2")
    title(ax[2], "c", "Corrections ranked by truth")
    fig.tight_layout()
    save(fig, "F3_virtual_species")


def fig_cv_inversion():
    e2 = pd.read_csv(R / "E02_reversal.csv").set_index("regime")
    k = pd.read_csv(R / "E09b_kba_cv_vs_truth.csv")
    t = pd.read_csv(R / "E17_cv_vs_truth_taiwan.csv")
    dis_truth = 62.0
    rows = [("disdat\n896 fits", e2.loc["random CV", "says_tgb_better_pct"], dis_truth),
            ("Kerala\n196 spp.", 100 * ((k.cv_tgb - k.cv_naive) > 0).mean(), 100 * ((k.kba_tgb - k.kba_naive) > 0).mean()),
            ("Taiwan\n131 spp.", 100 * ((t.cv_tgb - t.cv_naive) > 0).mean(), 100 * ((t.truth_tgb - t.truth_naive) > 0).mean())]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    x = np.arange(3)
    ax[0].bar(x - 0.19, [r[1] for r in rows], 0.36, color=ORANGE, label="random CV")
    ax[0].bar(x + 0.19, [r[2] for r in rows], 0.36, color=BLUE, label="independent survey")
    for xi, r in zip(x, rows):
        ax[0].text(xi - 0.19, r[1] + 1.5, f"{r[1]:.1f}%", ha="center", fontsize=6.5, color=INK)
        ax[0].text(xi + 0.19, r[2] + 1.5, f"{r[2]:.1f}%", ha="center", fontsize=6.5, color=INK)
    ax[0].set_xticks(x, [r[0] for r in rows], fontsize=6.8)
    ax[0].set_ylabel("% where bias correction wins")
    ax[0].set_ylim(0, 100)
    ax[0].legend(loc="upper left", ncol=2)
    title(ax[0], "a", "Random CV rejects the correction")
    for a_, D, cvn, cvt, tn, tt, nm in ((ax[1], k, "cv_naive", "cv_tgb", "kba_naive", "kba_tgb", "Kerala"),
                                       (ax[2], t, "cv_naive", "cv_tgb", "truth_naive", "truth_tgb", "Taiwan")):
        dx, dy = D[cvt] - D[cvn], D[tt] - D[tn]
        a_.scatter(dx, dy, s=7, c=BLUE, alpha=0.6, lw=0)
        a_.axhline(0, color=MUTED, lw=0.7)
        a_.axvline(0, color=MUTED, lw=0.7)
        a_.set_xlabel("random-CV AUC gain")
        a_.text(0.03, 0.03, f"survey: {(dy > 0).mean() * 100:.0f}% gain\nCV: {(dx > 0).mean() * 100:.1f}% gain",
                transform=a_.transAxes, ha="left", va="bottom", fontsize=6.5, color=INK)
        lim = max(abs(dy).max(), 0.05)
        a_.set_ylim(-lim * 1.05, lim * 1.05)
    ax[1].set_ylabel("survey-truth AUC gain")
    title(ax[1], "b", "Kerala, per species")
    title(ax[2], "c", "Taiwan, per species")
    fig.tight_layout()
    save(fig, "F4_cv_inversion")


def fig_cv_survey_size():
    e2 = pd.read_csv(R / "E02_reversal.csv").set_index("regime")
    k = pd.read_csv(R / "E09b_kba_cv_vs_truth.csv")
    t = pd.read_csv(R / "E17_cv_vs_truth_taiwan.csv")
    g = pd.read_csv(FIN / "results" / "glc24" / "summary_cv_agreement.csv")
    g = g[(g.choice == "TGB vs random, MLP") & (g.check == "auc_cv_vs_land")].iloc[0]
    rows = [("disdat\n896 fits", e2.loc["random CV", "says_tgb_better_pct"], 62.0),
            ("Kerala\n196 spp.", 100 * ((k.cv_tgb - k.cv_naive) > 0).mean(), 100 * ((k.kba_tgb - k.kba_naive) > 0).mean()),
            ("Taiwan\n131 spp.", 100 * ((t.cv_tgb - t.cv_naive) > 0).mean(), 100 * ((t.truth_tgb - t.truth_naive) > 0).mean()),
            ("Europe\n1,935 spp.", 100 * g.check_prefers_first, 100 * g.truth_prefers_first)]
    S = pd.read_csv(FIN / "results" / "survey_size_summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(rows))
    ax[0].bar(x - 0.19, [r[1] for r in rows], 0.36, color=ORANGE, label="random CV")
    ax[0].bar(x + 0.19, [r[2] for r in rows], 0.36, color=BLUE, label="independent survey")
    for xi, r in zip(x, rows):
        ax[0].text(xi - 0.19, r[1] + 1.5, f"{r[1]:.1f}", ha="center", fontsize=6.3, color=INK)
        ax[0].text(xi + 0.19, r[2] + 1.5, f"{r[2]:.1f}", ha="center", fontsize=6.3, color=INK)
    ax[0].set_xticks(x, [r[0] for r in rows], fontsize=6.6)
    ax[0].set_ylabel("% where bias correction wins")
    ax[0].set_ylim(0, 108)
    ax[0].legend(loc="upper left", ncol=2)
    title(ax[0], "a", "Random CV rejects the correction")
    for (sysn, sch), D in S.groupby(["system", "scheme"]):
        D = D[D.n_sites <= 320]
        ax[1].plot(D.n_sites, 100 * D.p_pick_corrected, color=BLUE if sysn == "Kerala" else INK,
                   ls="-" if sch == "random cells" else "--", marker="o" if sysn == "Kerala" else "s", ms=3, lw=1.4,
                   label=f"{sysn}, {'random' if sch == 'random cells' else 'clustered'} sites")
    ax[1].axhline(95, color=MUTED, lw=0.6, ls=":")
    ax[1].text(330, 91, "95%", fontsize=6.3, color=MUTED, ha="right", va="top")
    ax[1].scatter([5], [0], marker="X", s=40, color=ORANGE, zorder=4, clip_on=False)
    ax[1].text(7, 4, "random CV on all records: 0%", fontsize=6.5, color=ORANGE, va="bottom")
    ax[1].set_xscale("log")
    ax[1].set_xticks([5, 10, 20, 40, 80, 160, 320], ["5", "10", "20", "40", "80", "160", "320"])
    ax[1].set_ylim(-3, 103)
    ax[1].set_xlabel("independent survey sites used to judge")
    ax[1].set_ylabel("% of draws that pick the correction")
    ax[1].grid(axis="y", color=GRID, lw=0.6)
    ax[1].legend(loc="center right", fontsize=6.2)
    title(ax[1], "b", "How many survey sites are enough?")
    fig.tight_layout()
    save(fig, "S_cv_size")


def fig_glc_cnn():
    m = pd.read_csv(FIN / "results" / "glc24" / "summary_means.csv").set_index(["model", "bg"])
    combos = [("mlp", "random", "MLP, random bg", "#F0A25C"), ("cnn", "random", "CNN, random bg", ORANGE),
              ("mlp", "tgb", "MLP, target-group", "#8FB0D6"), ("cnn", "tgb", "CNN, target-group", BLUE)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [0.62, 1.38]})
    for i, (mo, bg, lb, c) in enumerate(combos):
        v = m.loc[(mo, bg), "auc_pa"]
        ax[0].bar(i, v, 0.7, color=c)
        ax[0].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=6.3, color=INK)
    ax[0].set_xticks(range(4), ["MLP\nrandom", "CNN\nrandom", "MLP\ntarget-\ngroup", "CNN\ntarget-\ngroup"], fontsize=6.3)
    ax[0].set_ylim(0.70, 0.83)
    ax[0].set_ylabel("mean survey AUC, 1,967 species")
    title(ax[0], "a", "Survey skill")
    layers = [("excess_roads", "roads"), ("excess_railways", "railways"), ("excess_lights", "night\nlights"),
              ("excess_built", "built-up"), ("excess_popdensity", "population"), ("excess_access_index", "access\nindex")]
    x = np.arange(len(layers))
    w = 0.2
    for i, (mo, bg, lb, c) in enumerate(combos):
        ax[1].bar(x + (i - 1.5) * w, [m.loc[(mo, bg), l] for l, _ in layers], w, color=c, label=lb)
    ax[1].set_xticks(x, [n for _, n in layers], fontsize=6.6)
    ax[1].axhline(0, color=MUTED, lw=0.7)
    ax[1].set_ylabel("access excess (SD), top-ranked\nplots minus true presences")
    ax[1].legend(loc="upper left", ncol=2, fontsize=6.3)
    ax[1].set_ylim(0, 0.78)
    ax[1].grid(axis="y", color=GRID, lw=0.6)
    title(ax[1], "b", "How close to people each model puts species")
    fig.tight_layout()
    save(fig, "S_glc_cnn")


def fig_missing_pairs():
    k9 = pd.read_csv(R / "E09b_kba_unrecorded_species_auc.csv")
    kn = pd.read_csv(R / "E11_neural_species_auc.csv").pivot_table(index="species", columns="loss", values="auc_missing")
    t17 = pd.read_csv(R / "E17_missing_pair_auc_taiwan.csv")
    items = [("nearest existing record", "dist", k9.nearest_record.mean(), t17.nearest_record.mean()),
             ("neural, random negatives", "naive", kn["an_ssdl"].mean(), t17.NN_an_ssdl.mean()),
             ("BRT, random background", "naive", k9.BRT_naive.mean(), t17.BRT_naive.mean()),
             ("neural, effort-aware likelihood", "corr", kn["occ_effort"].mean(), t17.NN_occ_effort.mean()),
             ("BRT, target-group background", "corr", k9.BRT_tgb.mean(), t17.BRT_tgb.mean())]
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    y = np.arange(len(items))[::-1]
    for yi, (nm, kind, kv, tv) in zip(y, items):
        ax.plot([kv, tv], [yi, yi], color=GRID, lw=2, zorder=1)
        ax.scatter(kv, yi, s=40, marker="o", color=BLUE, zorder=3, label="Kerala" if yi == y[0] else None)
        ax.scatter(tv, yi, s=40, marker="s", facecolor="white", edgecolor=BLUE, lw=1.4, zorder=3,
                   label="Taiwan" if yi == y[0] else None)
    ax.set_yticks(y, [i[0] for i in items])
    for lbl, it in zip(ax.get_yticklabels(), items):
        lbl.set_color(INK if it[1] == "corr" else MUTED)
        if it[1] == "corr":
            lbl.set_fontweight("bold")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("mean per-species AUC on pairs iNaturalist never recorded (survey truth)")
    ax.legend(loc="lower left")
    save(fig, "F5_core")


def draw_flip(ax, full_title=True):
    bb = pd.read_csv(FIN / "results" / "block_bootstrap_cis.csv")
    runs = [("Kerala\n<=2014", "E15_truth_source_summary_train2014.csv", "kba", "flip_kba_2014"),
            ("Kerala\n<=2017", "E15_truth_source_summary_train2017.csv", "kba", None),
            ("Taiwan\n<=2016", "E15_truth_source_summary_taiwan_train2016.csv", "twbbs", "flip_taiwan_2016"),
            ("Taiwan\n<=2018", "E15_truth_source_summary_taiwan_train2018.csv", "twbbs", "flip_taiwan_2018")]
    x, ticks = 0.0, []
    for name, f, lab, bkey in runs:
        S = pd.read_csv(R / f)
        for c, short in (("BRT_tgb - BRT_naive", "BRT"), ("NN_occ_effort - NN_an_ssdl", "neural")):
            for t, col, lb, bsuf in ((f"{lab}_same_cells", BLUE, "survey truth", "survey"),
                                     ("inat_later_same_cells", ORANGE, "later iNaturalist truth", "later_inat")):
                v = S[(S.truth == t) & (S.contrast == c)].delta.values[0]
                ax.bar(x, v, width=0.8, color=col, label=lb if x < 2 else None)
                if bkey:
                    r = bb[(bb.analysis == f"{bkey}_{bsuf}") & (bb.contrast == c)]
                    if len(r):
                        ax.errorbar(x, v, yerr=[[v - r.ci_lo.values[0]], [r.ci_hi.values[0] - v]], color=INK, lw=0.8,
                                    capsize=2)
                x += 1
            ticks.append((x - 1.5, f"{short}\n{name}"))
            x += 0.6
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=6.5)
    ax.set_ylabel("AUC gain of bias-corrected model")
    ax.set_ylim(-0.068, 0.088)
    ax.legend(loc="upper left", ncol=2)
    title(ax, "a" if full_title else "", "Same cell-species pairs, opposite verdicts (error bars: 95% spatial-block bootstrap)")


def draw_pfa(ax, letter="b"):
    occ = np.load(PROCESSED / "E16_atlas_occupancy.npz")
    K = systems.get("kba")
    pk = occ["pfa"][K["Y"] == 0]
    tw = pd.read_csv(R / "E17b_taiwan_occupancy_gate.csv").iloc[0]
    bins = np.linspace(0, 0.6, 31)
    ax.hist(pk, bins=bins, color=BLUE, weights=np.full(len(pk), 100 / len(pk)))
    ax.axvline(0.1, color=ORANGE, lw=1, ls="--")
    ax.text(0.105, ax.get_ylim()[1] * 0.85, f"{(pk > 0.1).mean() * 100:.0f}% of atlas absences\nhave P > 0.1\n"
            f"(Taiwan: {tw['frac_pfa_gt_0.1'] * 100:.1f}%)", fontsize=6.5, color=INK)
    ax.set_xlabel("P(false absence), occupancy model")
    ax.set_ylabel("% of atlas-absent pairs")
    title(ax, letter, "Atlas absences are imperfect")


def draw_robust(ax, letter="c"):
    Kc = pd.read_csv(R / "E16_rescored_contrasts.csv")
    Tc = pd.read_csv(R / "E17b_taiwan_occupancy_rescored.csv")
    its = [(f"Kerala: {v}", Kc[(Kc.variant == k) & (Kc.contrast == "BRT_tgb - BRT_naive")].delta.values[0])
           for k, v in (("atlas", "as given"), ("drop_pfa_gt_0.1", "false abs. dropped"), ("soft_occupancy", "soft labels"),
                        ("wet_only", "wet season"), ("dry_only", "dry season"))]
    its += [(f"Taiwan: {v}", Tc[(Tc.variant == k) & (Tc.contrast == "BRT_tgb - BRT_naive")].delta.values[0])
            for k, v in (("survey", "as given"), ("drop_pfa_gt_0.1", "false abs. dropped"))]
    yy = np.arange(len(its))[::-1]
    ax.barh(yy, [v for _, v in its], height=0.6, color=BLUE)
    for yv, (_, v) in zip(yy, its):
        ax.text(v + 0.001, yv, f"+{v:.3f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(yy, [n for n, _ in its], fontsize=6.5)
    ax.set_xlim(0, 0.056)
    ax.set_xlabel("AUC gain, BRT target-group - random")
    title(ax, letter, "Gain survives every truth variant")


def fig_flip_occupancy():
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.35)
    draw_flip(fig.add_subplot(gs[0, :]))
    draw_pfa(fig.add_subplot(gs[1, 0]))
    draw_robust(fig.add_subplot(gs[1, 1]))
    save(fig, "F6_flip_occupancy")


def fig_flip_only():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    draw_flip(ax, full_title=False)
    ax.set_title("")
    save(fig, "S_flip")


def fig_occ_only():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.6})
    draw_pfa(ax[0], "a")
    draw_robust(ax[1], "b")
    save(fig, "S_occupancy")


def fig_endemic_elevation():
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey="row")
    import statsmodels.formula.api as smf
    for row, (nm, f) in enumerate((("Kerala", "E15_endemic_gain_covariates.csv"),
                                   ("Taiwan", "E15_endemic_gain_covariates_taiwan.csv"))):
        E = pd.read_csv(R / f)
        for col, (v, xl) in enumerate((("exposure", "exposure: share of the species' survey cells\nthat iNaturalist never visited"),
                                       ("elev_mean", "mean elevation of the species' survey range (km)"))):
            a_ = ax[row, col]
            e = E.endemic == 1
            a_.scatter(E.loc[~e, v], E.loc[~e, "gain"], s=8, color=MUTED, alpha=0.55, lw=0, label="other species")
            a_.scatter(E.loc[e, v], E.loc[e, "gain"], s=16, color=ORANGE, lw=0,
                       label="Western Ghats endemic" if row == 0 else "Taiwan-concentrated")
            m = smf.ols(f"gain ~ {v}", E).fit(cov_type="HC3")
            xs = np.linspace(E[v].min(), E[v].max(), 20)
            a_.plot(xs, m.params.Intercept + m.params[v] * xs, color=BLUE, lw=1.6)
            a_.text(0.03, 0.96, f"bivariate slope {m.params[v]:+.3f}, p = {m.pvalues[v]:.1g}", transform=a_.transAxes,
                    va="top", fontsize=6.5, color=INK)
            a_.axhline(0, color=MUTED, lw=0.6)
            a_.set_xlabel(xl, fontsize=7)
            if col == 0:
                a_.set_ylabel(f"{nm}: AUC gain from\ntarget-group background")
            a_.legend(loc="lower right", fontsize=6.3)
    title(ax[0, 0], "a", "Kerala: gain tracks exposure")
    title(ax[0, 1], "b", "Kerala: gain tracks elevation")
    title(ax[1, 0], "c", "Taiwan: exposure does not explain it")
    title(ax[1, 1], "d", "Taiwan: elevation replicates")
    fig.tight_layout()
    save(fig, "F7_endemic_elevation")


def fig_discovery():
    d = pd.read_csv(R / "E05_full_scores_blockboot.csv")
    d = d[(d.group == "all") & (d.stratum == "all")]
    fig = plt.figure(figsize=(7.2, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.15)
    ax = fig.add_subplot(gs[0])
    ms = [("bayes", "Bayesian discovery model", BLUE, "o"), ("bayes_intercepts_only", "same, intercepts only", BLUE, "s"),
          ("effort_glm_covs", "effort regression + habitat", MUTED, "^")]
    for i, (m, lb, c, mk) in enumerate(ms):
        g = d[d.model == m].sort_values("origin")
        xo = g.origin + (i - 1) * 0.25
        ax.errorbar(xo, g.delta_vs_effort_glm, yerr=[g.delta_vs_effort_glm - g.ci_lo, g.ci_hi - g.delta_vs_effort_glm],
                    fmt=mk, color=c, mfc=c if mk == "o" else "white", ms=5, lw=1, capsize=2, label=lb)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks([2016, 2018, 2020, 2022])
    ax.set_xlabel("forecast origin (fit on records up to this year, score the next 3 years)")
    ax.set_ylabel("log-score gain over effort\nregression (nats per 5 km cell)")
    ax.legend(loc="upper right")
    title(ax, "a", "Forecasting new species-in-cell records")
    ax = fig.add_subplot(gs[1])
    with rasterio.open(PROCESSED / "E05_expected_new_per_10_visits.tif") as r:
        v = r.read(1).astype(float)
        nod = r.nodata
        b = r.bounds
    if nod is not None:
        v[v == nod] = np.nan
    im = ax.imshow(v, cmap=BRAMP, extent=[b.left, b.right, b.bottom, b.top], interpolation="nearest")
    mapax(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01, shrink=0.7)
    cb.set_label("expected new species-in-cell\nrecords from next 10 visits", fontsize=7)
    title(ax, "b", "Survey-priority surface (to 2026)")
    save(fig, "F8_discovery")


if __name__ == "__main__":
    which = sys.argv[1:] or ["study", "bias", "virtual", "cv", "missing", "flip", "endemic", "discovery"]
    fns = {"study": fig_study_systems, "bias": fig_bias_anatomy, "virtual": fig_virtual, "cv": fig_cv_inversion,
           "missing": fig_missing_pairs, "flip": fig_flip_occupancy, "endemic": fig_endemic_elevation,
           "discovery": fig_discovery, "sflip": fig_flip_only, "socc": fig_occ_only,
           "cvsize": fig_cv_survey_size, "glc": fig_glc_cnn}
    for w in which:
        fns[w]()
