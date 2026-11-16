import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, "experiments")
from E02_disdat_benchmark import design_matrix
from evst.data import disdat as d
from evst.paths import RESULTS
from evst.validate.metrics import auc

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--regions", default=",".join(d.REGIONS))
ap.add_argument("--rff", type=int, default=150)
ap.add_argument("--ls", type=float, default=0.1)
ap.add_argument("--tau", type=float, default=1.0)
ap.add_argument("--tag", default="pilot")
a = ap.parse_args()
dev = "cuda"
META = {"siteid", "spid", "x", "y", "occ", "group"}


def fit(Fp, Fb, Gp, Gb, pres_idx, species_bg, ridge=1.0, tau=None, steps=300):
    S = len(pres_idx)
    k = Fp.shape[1]
    B = torch.zeros(k, S, device=dev, requires_grad=True)
    al = torch.zeros(S, device=dev, requires_grad=True)
    params = [B, al]
    wg = None
    if tau is not None:
        wg = torch.zeros(Gp.shape[1], device=dev, requires_grad=True)
        params.append(wg)
    opt = torch.optim.LBFGS(params, lr=1, max_iter=steps, line_search_fn="strong_wolfe")
    sp_of_p = torch.cat([torch.full((len(ix),), s, device=dev) for s, ix in enumerate(pres_idx)])
    pidx = torch.cat(pres_idx)

    def closure():
        opt.zero_grad()
        zp = (Fp[pidx] * B[:, sp_of_p].T).sum(1) + al[sp_of_p]
        if wg is not None:
            zp = zp + Gp[pidx] @ wg
        loss = torch.nn.functional.softplus(-zp).sum()
        for s in range(S):
            Fb_s, Gb_s, wts = species_bg[s]
            zb = Fb_s @ B[:, s] + al[s]
            if wg is not None:
                zb = zb + Gb_s @ wg
            loss = loss + (wts * torch.nn.functional.softplus(zb)).sum()
        loss = loss + (B ** 2).sum() / (2 * ridge ** 2)
        if wg is not None:
            loss = loss + (wg ** 2).sum() / (2 * tau ** 2)
        loss.backward()
        return loss

    opt.step(closure)
    return B.detach(), al.detach(), (wg.detach() if wg is not None else None)


rows = []
for r in a.regions.split(","):
    po, bg = d.load_po(r), d.load_bg(r)
    cols = [c for c in po.columns if c not in META]
    cats = d.CATEGORICAL[r]
    tests = {g: d.load_pa(r, g) for g in d.GROUPS[r]}
    allv = pd.concat([po[cols], bg[cols]] + [e[cols] for _, e in tests.values()])
    levels = {c: sorted(allv[c].dropna().unique().tolist()) for c in cats}
    sites = po.drop_duplicates(["x", "y"])[["x", "y"] + cols].reset_index(drop=True)
    key = {xy: i for i, xy in enumerate(zip(sites.x, sites.y))}
    raw_s = design_matrix(sites, cols, cats, levels)
    raw_b = design_matrix(bg, cols, cats, levels)
    sc = StandardScaler().fit(np.r_[raw_s, raw_b])
    poly = PolynomialFeatures(2, include_bias=False).fit(sc.transform(raw_b))
    sc2 = StandardScaler().fit(poly.transform(sc.transform(np.r_[raw_s, raw_b])))
    feat = lambda X: sc2.transform(poly.transform(sc.transform(X)))
    xy_all = np.r_[sites[["x", "y"]].values, bg[["x", "y"]].values]
    ext = np.ptp(xy_all, 0).max()
    rng = np.random.default_rng(0)
    Om = rng.normal(0, 1 / (a.ls * ext), (2, a.rff))
    ph = rng.uniform(0, 2 * np.pi, a.rff)
    rff = lambda xy: np.sqrt(2 / a.rff) * np.cos(xy @ Om + ph)
    T = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)
    Fp, Fb = T(feat(raw_s)), T(feat(raw_b))
    Gp, Gb = T(rff(sites[["x", "y"]].values)), T(rff(bg[["x", "y"]].values))
    sps = sorted(po.spid.unique())
    pres_idx, grp_of = [], {}
    for sp in sps:
        p = po[po.spid == sp].drop_duplicates(["x", "y"])
        pres_idx.append(torch.as_tensor([key[xy] for xy in zip(p.x, p.y)], device=dev))
        grp_of[sp] = p.group.iloc[0]
    tg_idx = {}
    for sp in sps:
        g = po[po.group == grp_of[sp]] if po.group.nunique() > 1 else po
        g = g.drop_duplicates(["x", "y"])
        tg_idx[sp] = torch.as_tensor([key[xy] for xy in zip(g.x, g.y)], device=dev)

    def bgsets(kind):
        out = []
        for s, sp in enumerate(sps):
            n = len(pres_idx[s])
            if kind == "random":
                out.append((Fb, Gb, torch.full((Fb.shape[0],), n / Fb.shape[0], device=dev)))
            else:
                ti = tg_idx[sp]
                out.append((Fp[ti], Gp[ti], torch.full((len(ti),), n / len(ti), device=dev)))
        return out

    variants = {"glm_random": ("random", None), "glm_tgb": ("tgb", None),
                "jointbias_random": ("random", a.tau), "jointbias_tgb": ("tgb", a.tau)}
    preds = {}
    for vname, (kind, tau) in variants.items():
        B, al, wg = fit(Fp, Fb, Gp, Gb, pres_idx, bgsets(kind), tau=tau)
        preds[vname] = (B, al)
    for s, sp in enumerate(sps):
        grp = grp_of[sp]
        pa, env = tests[grp if grp in tests else None]
        if sp not in pa.columns:
            continue
        Fe = T(feat(design_matrix(env, cols, cats, levels)))
        y = pa[sp].to_numpy()
        row = {"region": r, "species": sp, "n_pres": len(pres_idx[s])}
        for vname, (B, al) in preds.items():
            row[vname] = auc(y, (Fe @ B[:, s] + al[s]).cpu().numpy())
        rows.append(row)
    df = pd.DataFrame(rows)
    print(r, df[df.region == r][list(variants)].mean().round(4).to_dict(), flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / f"E07_jointbias_{a.tag}.csv", index=False)
print("\nmean PA AUC:", df[list(variants)].mean().round(4).to_dict())
for v in ("jointbias_random", "jointbias_tgb"):
    for ref in ("glm_random", "glm_tgb"):
        dd = (df[v] - df[ref]).dropna()
        print(f"{v} - {ref}: mean {dd.mean():+.4f}, better {100*(dd>0).mean():.0f}%, wilcoxon p={wilcoxon(dd).pvalue:.2e}")
