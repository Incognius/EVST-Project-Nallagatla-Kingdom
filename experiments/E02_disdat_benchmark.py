from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from evst.data import disdat as d
from evst.models.sdm import MODELS
from evst.paths import RESULTS
from evst.validate.blocks import autocorr_range, random_folds, spatial_folds
from evst.validate.metrics import auc, cor

warnings.filterwarnings("ignore")
META = {"siteid", "spid", "x", "y", "occ", "group"}


def design_matrix(df: pd.DataFrame, cols: list[str], cats: list[str], levels: dict) -> np.ndarray:
    num = [c for c in cols if c not in cats]
    parts = [df[num].to_numpy(float)]
    for c in cats:
        parts.append((df[c].to_numpy()[:, None] == np.array(levels[c])[None, :]).astype(float))
    return np.hstack(parts)


def region_setup(r: str):
    po, bg = d.load_po(r), d.load_bg(r)
    cols = [c for c in po.columns if c not in META]
    cats = d.CATEGORICAL[r]
    tests = {g: d.load_pa(r, g) for g in d.GROUPS[r]}
    allv = pd.concat([po[cols], bg[cols]] + [e[cols] for _, e in tests.values()])
    levels = {c: sorted(allv[c].dropna().unique().tolist()) for c in cats}
    num = [c for c in cols if c not in cats]
    ranges = [autocorr_range(bg.x.values, bg.y.values, bg[c].values.astype(float)) for c in num]
    size_raw = float(np.nanmedian(ranges))
    ext = min(np.ptp(bg.x.values), np.ptp(bg.y.values))
    size = min(size_raw, ext / 5)
    return po, bg, cols, cats, levels, tests, size, ranges


def run_species(r, sp, po, bg, cols, cats, levels, tests, size, models, seed=0):
    rows = []
    p = po[po.spid == sp].drop_duplicates(["x", "y"])
    grp = p.group.iloc[0]
    tg = po[po.group == grp] if po.group.nunique() > 1 else po
    tg = tg.drop_duplicates(["x", "y"])
    pidx = {xy: i for i, xy in enumerate(zip(p.x, p.y))}
    tg_pres = np.array([pidx.get(xy, -1) for xy in zip(tg.x, tg.y)])
    own = set(zip(p.x, p.y))
    tg_other = tg[[xy not in own for xy in zip(tg.x, tg.y)]]
    pa, env = tests[grp if grp in tests else None]
    if sp not in pa.columns:
        return rows
    Xp, Xb = design_matrix(p, cols, cats, levels), design_matrix(bg, cols, cats, levels)
    Xtg, Xtgo = design_matrix(tg, cols, cats, levels), design_matrix(tg_other, cols, cats, levels)
    Xpa = design_matrix(env, cols, cats, levels)
    ypa = pa[sp].to_numpy()

    fr_p, fr_b = random_folds(len(p), 5, seed), random_folds(len(bg), 5, seed + 1)
    xy_all = np.r_[p[["x", "y"]].values, bg[["x", "y"]].values, tg[["x", "y"]].values, tg_other[["x", "y"]].values]
    fs_all = spatial_folds(xy_all[:, 0], xy_all[:, 1], size, 5, seed)
    n1, n2, n3 = len(p), len(p) + len(bg), len(p) + len(bg) + len(tg)
    fs_p, fs_b, fs_tg, fs_tgo = fs_all[:n1], fs_all[n1:n2], fs_all[n2:n3], fs_all[n3:]

    for mname in models:
        for bgname in ("random", "tgb"):
            Xbg_full = Xb if bgname == "random" else Xtg
            t0 = time.time()
            m = MODELS[mname]().fit(np.r_[Xp, Xbg_full], np.r_[np.ones(len(Xp)), np.zeros(len(Xbg_full))])
            s_pa = m.predict(Xpa)
            res = {"region": r, "group": grp, "species": sp, "n_pres": len(p), "model": mname, "bg": bgname,
                   "pa_auc": auc(ypa, s_pa), "pa_cor": cor(ypa, s_pa), "pa_prev": ypa.mean()}

            for scheme, fp, fb, ftg in (("cv_rand", fr_p, fr_b, None), ("cv_spat", fs_p, fs_b, fs_tg)):
                aucs, aucs_tg = [], []
                for k in range(5):
                    tr_p, te_p = fp != k, fp == k
                    if te_p.sum() == 0 or tr_p.sum() < 3:
                        continue
                    tr_b, te_b = fb != k, fb == k
                    if bgname == "random":
                        Xbg_tr = Xb[tr_b]
                    else:
                        held = np.zeros(len(tg), bool)
                        held[tg_pres >= 0] = te_p[tg_pres[tg_pres >= 0]]
                        if ftg is not None:
                            held |= ftg == k
                        Xbg_tr = Xtg[~held]
                    mk = MODELS[mname]().fit(np.r_[Xp[tr_p], Xbg_tr],
                                             np.r_[np.ones(tr_p.sum()), np.zeros(len(Xbg_tr))])
                    sp_te, sb_te = mk.predict(Xp[te_p]), mk.predict(Xb[te_b])
                    aucs.append(auc(np.r_[np.ones(len(sp_te)), np.zeros(len(sb_te))], np.r_[sp_te, sb_te]))
                    if scheme == "cv_spat":
                        te_tg = fs_tgo == k
                        if te_tg.sum() > 0:
                            st = mk.predict(Xtgo[te_tg])
                            aucs_tg.append(auc(np.r_[np.ones(len(sp_te)), np.zeros(len(st))], np.r_[sp_te, st]))
                res[f"{scheme}_auc"] = float(np.nanmean(aucs)) if aucs else np.nan
                if scheme == "cv_spat":
                    res["cv_spat_tg_auc"] = float(np.nanmean(aucs_tg)) if aucs_tg else np.nan
            res["secs"] = time.time() - t0
            rows.append(res)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="GLM,BRT,RF,MaxEnt")
    ap.add_argument("--regions", default=",".join(d.REGIONS))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default="E02_disdat.csv")
    ap.add_argument("--block-source", default="python", choices=["python", "blockCV"],
                    help="block size from our variogram (capped) or from R blockCV ranges (median, capped)")
    ap.add_argument("--block-mult", type=float, default=1.0)
    a = ap.parse_args()
    models = a.models.split(",")
    out_csv = RESULTS / a.out
    prev = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["region"])
    allrows, meta = prev.to_dict("records"), []
    for r in a.regions.split(","):
        if r in set(prev.region):
            continue
        t = time.time()
        po, bg, cols, cats, levels, tests, size, ranges = region_setup(r)
        if a.block_source == "blockCV":
            rb = pd.read_csv(RESULTS / "R_blockCV_ranges.csv")
            rr = rb[rb.region == r].range_blockCV.median()
            if r in ("CAN", "NSW", "SA"):
                rr = rr / 111_000
            ext = min(np.ptp(bg.x.values), np.ptp(bg.y.values))
            size = min(rr, ext / 5)
        size *= a.block_mult
        meta.append({"region": r, "block_size": size, "range_median": float(np.nanmedian(ranges)), "ranges": ranges})
        sps = sorted(po.spid.unique())
        out = Parallel(n_jobs=a.jobs)(delayed(run_species)(r, sp, po, bg, cols, cats, levels, tests, size, models)
                                      for sp in sps)
        rows = [x for o in out for x in o]
        allrows += rows
        print(f"{r}: {len(sps)} species, block={size:.3g}, {time.time()-t:.0f}s", flush=True)
        pd.DataFrame(allrows).to_csv(RESULTS / a.out, index=False)
    mf = RESULTS / a.out.replace(".csv", "_blocks.csv")
    old = pd.read_csv(mf) if mf.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(meta)]).to_csv(mf, index=False)


if __name__ == "__main__":
    main()
