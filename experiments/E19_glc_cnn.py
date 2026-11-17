"""E19: does spatial context (patch CNN) help on real survey truth, and does it absorb more observer bias?

GeoLifeCLEF 2024 (Europe, plants). Trained on presence-only records only; the ~89k presence/absence survey plots
are never seen in training and serve as truth. This is the "fair test of the CNN" promised at mid-evaluation:
E03's virtual species responded to point values only, so spatial context could not help there.

One multi-species network per run (shared trunk, one output per species):
  --model mlp   point model on the 1 km centre pixel
  --model cnn   patch CNN on a P x P km window of the same layers (+ the centre pixel)
  --bg random   positives at PO cells vs random land cells          (assume-negative, uncorrected)
  --bg tgb      positives vs other species' PO cells (target group)  (bias-corrected, Phillips 2009)
  --split random | block   20% of PO cells held out at random or in 50 km blocks (internal CV)

Scores written per species: PA-truth AUC, internal CV AUCs, and the "access excess": the mean access z-score
(roads, population, night lights, built-up) of the model's top-k PA sites minus that of the k true presences.
Inputs (environment) never include the access layers; they are used only to measure leakage.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from evst.paths import INTERIM, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["mlp", "cnn"], required=True)
ap.add_argument("--bg", choices=["random", "tgb"], required=True)
ap.add_argument("--split", choices=["random", "block"], default="random")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--patch", type=int, default=15)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--tag", default="")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
dev = a.device
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
G = INTERIM / "glc24"
meta = json.loads((G / "meta.json").read_text())
H, W = meta["shape"]
t0 = time.time()

# ---------------------------------------------------------------- data
stack = np.load(G / "stack.npy", mmap_mode="r")  # (C, H, W) float16, standardised, NaN -> 0
C = stack.shape[0]
r = a.patch // 2
S_gpu = torch.zeros((C, H + 2 * r, W + 2 * r), dtype=torch.float16, device=dev)
for c in range(C):
    S_gpu[c, r:r + H, r:r + W] = torch.from_numpy(np.ascontiguousarray(stack[c]))
land = np.load(G / "land_cells.npy")
po = np.load(G / "po_cells.npz")  # cell (n,), sp (n,) for records of modelled species; cells_all = every PO cell
pa = np.load(G / "pa.npz")  # cell (n_survey,), Y (n_survey, S) uint8, access (n_survey, k) float32
species = np.load(G / "species.npy")
S = len(species)
print(f"stack {stack.shape}, land cells {len(land):,}, species {S}, load {time.time() - t0:.0f}s", flush=True)

cells_all = po["cells_all"]  # unique cells holding any PO record (all species): the target group
xy = np.column_stack([cells_all % W, cells_all // W])
if a.split == "random":
    hold = rng.random(len(cells_all)) < 0.2
else:
    blk = (xy // 50).astype(np.int64)
    bid = blk[:, 0] * 100000 + blk[:, 1]
    ub = np.unique(bid)
    hold = np.isin(bid, ub[rng.random(len(ub)) < 0.2])
cpos = pd.Series(np.arange(len(cells_all)), index=cells_all)
rec_ci = cpos[po["cell"]].to_numpy()
order = np.lexsort((po["sp"], rec_ci))
rec_ci, rec_sp = rec_ci[order], po["sp"][order]
pairs = np.unique(rec_ci.astype(np.int64) * S + rec_sp)
rec_ci, rec_sp = (pairs // S).astype(np.int64), (pairs % S).astype(np.int64)
indptr = np.zeros(len(cells_all) + 1, np.int64)
np.add.at(indptr, rec_ci + 1, 1)
indptr = np.cumsum(indptr)
train_ci = np.flatnonzero(~hold)
test_ci = np.flatnonzero(hold)
print(f"PO cells {len(cells_all):,} (held out {hold.mean():.1%}), modelled pairs {len(pairs):,}", flush=True)

indptr_t = torch.tensor(indptr, device=dev)
sp_t = torch.tensor(rec_sp, device=dev)
cells_t = torch.tensor(cells_all, device=dev)
train_t = torch.tensor(train_ci, device=dev)
land_t = torch.tensor(land, device=dev)
off = torch.arange(a.patch, device=dev)


def labels(ci):
    """Dense (B, S) labels for PO-cell indices ci from the CSR."""
    st, en = indptr_t[ci], indptr_t[ci + 1]
    n = en - st
    rows = torch.repeat_interleave(torch.arange(len(ci), device=dev), n)
    idx = torch.repeat_interleave(st - torch.cumsum(n, 0) + n, n) + torch.arange(int(n.sum()), device=dev)
    Y = torch.zeros(len(ci), S, device=dev)
    Y[rows, sp_t[idx]] = 1
    return Y


def patches(cells):
    rr = (cells // W).view(-1, 1, 1) + off.view(1, -1, 1)
    cc = (cells % W).view(-1, 1, 1) + off.view(1, 1, -1)
    return S_gpu[:, rr, cc].permute(1, 0, 2, 3).float()  # (B, C, P, P)


def centre(cells):
    return S_gpu[:, cells // W + r, cells % W + r].T.float()


# ---------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(C, 512), nn.SiLU(), nn.Dropout(0.2), nn.Linear(512, 512), nn.SiLU(),
                               nn.Dropout(0.2), nn.Linear(512, 512), nn.SiLU(), nn.Dropout(0.2))
        self.head = nn.Linear(512, S)

    def forward(self, cells):
        return self.head(self.f(centre(cells)))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        cb = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.SiLU()]
        self.conv = nn.Sequential(*cb(C, 64), *cb(64, 64), nn.MaxPool2d(2), *cb(64, 128), *cb(128, 128),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.f = nn.Sequential(nn.Linear(128 + C, 512), nn.SiLU(), nn.Dropout(0.2), nn.Linear(512, 512), nn.SiLU(),
                               nn.Dropout(0.2))
        self.head = nn.Linear(512, S)

    def forward(self, cells):
        x = patches(cells)
        if self.training:  # rotations/flips: orientation of the landscape carries no information
            k = int(torch.randint(0, 4, (1,)))
            x = torch.rot90(x, k, (2, 3))
            if torch.rand(1).item() < 0.5:
                x = x.flip(-1)
        return self.head(self.f(torch.cat([self.conv(x), x[:, :, r, r]], 1)))


net = (MLP() if a.model == "mlp" else CNN()).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)


def bal_bce(logit, y, mask):
    pos, neg = y * mask, (1 - y) * mask
    lp = -(F.logsigmoid(logit) * pos).sum(0) / pos.sum(0).clamp(min=1)
    ln = -(F.logsigmoid(-logit) * neg).sum(0) / neg.sum(0).clamp(min=1)
    has = pos.sum(0) > 0
    return lp[has].mean() + ln.mean()


net.train()
for step in range(a.steps):
    ci = train_t[torch.randint(0, len(train_t), (a.batch,), device=dev)]
    Y = labels(ci)
    with torch.autocast(dev, dtype=torch.bfloat16):
        if a.bg == "tgb":
            loss = bal_bce(net(cells_t[ci]).float(), Y, torch.ones_like(Y))
        else:
            rc = land_t[torch.randint(0, len(land_t), (a.batch,), device=dev)]
            logit = net(torch.cat([cells_t[ci], rc])).float()
            Yr = torch.zeros_like(Y)
            loss = bal_bce(logit, torch.cat([Y, Yr]), torch.cat([Y, torch.ones_like(Yr)]))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if step % 2000 == 0:
        print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
net.eval()


@torch.no_grad()
def predict(cells, chunk=4096):
    out = []
    for i in range(0, len(cells), chunk):
        with torch.autocast(dev, dtype=torch.bfloat16):
            out.append(net(torch.as_tensor(cells[i:i + chunk], device=dev)).float().cpu())
    return torch.cat(out).numpy()


def auc_cols(Y, P):
    """Per-column AUC (Mann-Whitney); Y (n,S) 0/1, P (n,S)."""
    R = pd.DataFrame(P).rank().to_numpy()
    n1 = Y.sum(0)
    n0 = len(Y) - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((R * Y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------- evaluation
res = pd.DataFrame({"species": species})
Ypa = pa["Y"].astype(np.float32)
Ppa = predict(pa["cell"])
res["auc_pa"] = auc_cols(Ypa, Ppa)
res["n_pa_pres"] = Ypa.sum(0)
# access excess: top-k predicted PA sites vs the k true presence sites (k = number of presences)
Az = pa["access"]
for j, name in enumerate(meta["access_names"] + ["access_index"]):
    a_j = Az[:, j] if j < Az.shape[1] else Az.mean(1)
    k = Ypa.sum(0).astype(int)
    top = np.argsort(-Ppa, axis=0)
    ex = np.array([a_j[top[:k[s], s]].mean() - a_j[Ypa[:, s] == 1].mean() if k[s] > 0 else np.nan for s in range(S)])
    res[f"excess_{name}"] = ex
# internal CV on held-out PO cells: positives = held-out cells with the species;
# negatives = (i) random land cells (the usual check), (ii) held-out PO cells without it (target-group check)
neg_land = land[rng.choice(len(land), 20000, replace=False)]
P_te, P_land = predict(cells_all[test_ci]), predict(neg_land)
Y_te = labels(torch.tensor(test_ci, device=dev)).cpu().numpy()
n_te = Y_te.sum(0)
res["n_cv_pres"] = n_te
cv_rand = np.full(S, np.nan)
for s in np.flatnonzero(n_te >= 5):
    p1 = P_te[Y_te[:, s] == 1, s]
    y = np.r_[np.ones(len(p1)), np.zeros(len(P_land))]
    cv_rand[s] = auc_cols(y[:, None], np.r_[p1, P_land[:, s]][:, None])[0]
res["auc_cv_vs_land"] = cv_rand
res["auc_cv_vs_tg"] = np.where(n_te >= 5, auc_cols(Y_te, P_te), np.nan)
name = f"E19_{a.model}_{a.bg}_{a.split}_s{a.seed}{a.tag}"
(RESULTS / "E19").mkdir(exist_ok=True)
res.to_csv(RESULTS / "E19" / f"{name}.csv", index=False)
ok = res.n_pa_pres >= 10
print(f"{name}: PA AUC {res.auc_pa[ok].mean():.4f} | CV vs land {np.nanmean(cv_rand):.4f} | CV vs TG "
      f"{res.auc_cv_vs_tg.mean():.4f} | access excess {res.excess_access_index[ok].mean():+.3f} | {time.time() - t0:.0f}s",
      flush=True)
