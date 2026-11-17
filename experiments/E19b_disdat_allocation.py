import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from evst.allocate import acquire
from evst.data import disdat
from evst.paths import RESULTS
from evst.validate.metrics import auc

ap = argparse.ArgumentParser()
ap.add_argument("--regions", default="CAN,NZ,SWI")
ap.add_argument("--splits", type=int, default=10)
ap.add_argument("--budgets", default="0,10,20,40,80,160,320")
ap.add_argument("--pool", type=int, default=2000)
ap.add_argument("--test", type=int, default=3000)
a = ap.parse_args()
budgets = [int(x) for x in a.budgets.split(",")]
strategies = acquire.BASE + acquire.NEW + acquire.ORACLE
dev = "cuda"


class Net(torch.nn.Module):
    def __init__(self, d, m, h=128, drop=0.2):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.SiLU(), torch.nn.Dropout(drop), torch.nn.Linear(h, h),
                                     torch.nn.SiLU(), torch.nn.Dropout(drop), torch.nn.Linear(h, m))

    def forward(self, x):
        return self.f(x)


def train(Xpo, Ypo, Xpa=None, Ypa=None, steps=1500, seed=0, init=None, pa_w=1.0):
    torch.manual_seed(seed)
    net = Net(Xpo.shape[1], Ypo.shape[1]).to(dev)
    if init is not None:
        net.load_state_dict(init)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = ((1 - Ypo).sum(0) / Ypo.sum(0).clamp(min=1)).clamp(max=50)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(net(Xpo), Ypo, pos_weight=pw)
        if Xpa is not None:
            loss = loss + pa_w * F.binary_cross_entropy_with_logits(net(Xpa), Ypa)
        loss.backward()
        opt.step()
    return net


def predict(net, X, train_mode=False):
    net.train(train_mode)
    with torch.no_grad():
        p = torch.sigmoid(net(X)).cpu().numpy()
    net.train()
    return p


rows = []
for region in a.regions.split(","):
    t_r = time.time()
    po = disdat.load_po(region)
    pa, env = disdat.load_pa(region, None)
    cols = [c for c in env.columns if c not in ("group", "siteid", "x", "y") and c not in disdat.CATEGORICAL[region]]
    mu, sd = po[cols].mean(), po[cols].std() + 1e-9
    sites = po.drop_duplicates("siteid")
    spp = [c for c in pa.columns[4:]]
    spos = pd.Series(np.arange(len(spp)), index=spp)
    Ypo = np.zeros((len(sites), len(spp)), np.float32)
    sp_of = po[po.spid.isin(spp)]
    srow = pd.Series(np.arange(len(sites)), index=sites.siteid)
    Ypo[srow[sp_of.siteid].to_numpy(), spos[sp_of.spid].to_numpy()] = 1
    Xpo = torch.as_tensor(((sites[cols] - mu) / sd).to_numpy(np.float32), device=dev)
    Ypo_t = torch.as_tensor(Ypo, device=dev)
    Xall = ((env[cols] - mu) / sd).to_numpy(np.float32)
    Yall = pa[spp].to_numpy().astype(np.int8)
    xy = env[["x", "y"]].to_numpy(float)
    ext = np.ptp(xy, 0).max()
    near = cKDTree(sites[["x", "y"]].to_numpy(float)).query_ball_point(xy, r=0.005 * ext, return_length=True)
    base = train(Xpo, Ypo_t)
    init = {k: v.clone() for k, v in base.state_dict().items()}
    Xt_all = torch.as_tensor(Xall, device=dev)
    psi_all0 = predict(base, Xt_all)
    blk = (np.floor((xy[:, 0] - xy[:, 0].min()) / (ext / 10)) * 100 + np.floor((xy[:, 1] - xy[:, 1].min()) / (ext / 10))).astype(int)
    u, s, _ = np.linalg.svd(Xall - Xall.mean(0), full_matrices=False)
    Ze = u[:, :5] * s[:5]
    feat_all = np.column_stack([(Ze - Ze.mean(0)) / Ze.std(0), (xy - xy.mean(0)) / xy.std(0)])
    for split in range(a.splits):
        rng = np.random.default_rng(split)
        ub = np.unique(blk)
        tb = rng.choice(ub, int(0.3 * len(ub)), replace=False)
        te_all = np.flatnonzero(np.isin(blk, tb))
        po_all = np.setdiff1d(np.arange(len(env)), te_all)
        test = rng.choice(te_all, min(a.test, len(te_all)), replace=False)
        pool_idx = rng.choice(po_all, min(a.pool, len(po_all)), replace=False)
        sub = np.r_[pool_idx]
        Yp, Up = Yall[sub], near[sub].astype(float)
        Xp_t = torch.as_tensor(Xall[sub], device=dev)
        Xte = torch.as_tensor(Xall[test], device=dev)

        def evaluate(net):
            p = predict(net, Xte)
            v = [auc(Yall[test, j], p[:, j]) for j in range(len(spp)) if 5 <= Yall[test, j].sum() <= len(test) - 5]
            return float(np.mean(v)), len(v)

        ref0 = train(Xpo, Ypo_t, steps=300, seed=split, init=init)
        a0, nsp = evaluate(ref0)
        for strat in strategies:
            t0 = time.time()
            srng = np.random.default_rng(100 + split)
            revealed, net = [], base
            psi = psi_all0[sub]
            ctx = {"R": np.zeros_like(Yp), "U": Up, "cxy": xy[sub], "feat": feat_all[sub], "Y": Yp, "min_sep": 0.02 * ext}
            rows.append({"region": region, "split": split, "strategy": strat, "n_sites": 0, "auc": a0, "n_species": nsp,
                         "revealed_low_effort_frac": np.nan, "prevalence_revealed": np.nan})
            local = np.arange(len(sub))
            for nb in budgets[1:]:
                ctx.update(psi=psi, sample=(lambda n=net: predict(n, Xp_t, True)))
                revealed += acquire.choose(strat, local, revealed, nb - len(revealed), ctx, srng)
                rv = np.array(revealed)
                net = train(Xpo, Ypo_t, Xp_t[rv], torch.as_tensor(Yp[rv].astype(np.float32), device=dev), steps=300,
                            seed=split, init=init)
                new = predict(net, Xp_t)
                if strat == "adaptive":
                    acquire.update_lambda(ctx, new, psi)
                psi = new
                aa, nsp = evaluate(net)
                rows.append({"region": region, "split": split, "strategy": strat, "n_sites": nb, "auc": aa, "n_species": nsp,
                             "revealed_low_effort_frac": float((Up[rv] == 0).mean()),
                             "prevalence_revealed": float(Yp[rv].mean())})
        print(region, split, round(time.time() - t_r), "s", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "E19b_disdat_allocation.csv", index=False)

D = pd.DataFrame(rows)
print(D.groupby(["region", "strategy", "n_sites"]).auc.mean().unstack("n_sites").round(4).to_string())
