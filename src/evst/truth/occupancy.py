from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from evst.data import kba

SEASONS = ("Wet", "Dry")


def histories(S: dict, kmax: int = 16) -> dict:
    atlas, _ = kba._atlas()
    atlas = atlas[atlas["Cell.ID"].isin(S["ev"])]
    cpos = pd.Series(np.arange(len(S["ev"])), index=S["ev"])
    spos = pd.Series(np.arange(len(S["cand"])), index=S["cand"])
    out = {}
    for s in SEASONS:
        a = atlas[atlas.Season == s]
        lists = a.drop_duplicates("List.ID")[["List.ID", "Cell.ID", "n.observers", "Time"]].copy()
        lists["k"] = lists.groupby("Cell.ID").cumcount()
        lists = lists[lists.k < kmax]
        hour = pd.to_numeric(lists.Time.str.split(":").str[0], errors="coerce").fillna(9)
        n, m = len(S["ev"]), len(S["cand"])
        y = np.zeros((n, kmax, m), np.float32)
        mask = np.zeros((n, kmax), np.float32)
        obs = np.zeros((n, kmax), np.float32)
        morn = np.zeros((n, kmax), np.float32)
        ci = cpos[lists["Cell.ID"]].to_numpy()
        ki = lists.k.to_numpy()
        mask[ci, ki] = 1
        obs[ci, ki] = np.log(lists["n.observers"].clip(1, 20).to_numpy())
        morn[ci, ki] = (hour < 10).astype(np.float32).to_numpy()
        lk = lists.set_index("List.ID")
        d = a[a["Scientific.Name"].isin(spos.index) & a["List.ID"].isin(lk.index)]
        y[cpos[d["Cell.ID"]].to_numpy(), lk.loc[d["List.ID"], "k"].to_numpy(), spos[d["Scientific.Name"]].to_numpy()] = 1
        out[s] = dict(y=y, mask=mask, obs=obs, morn=morn)
    return out


def env_pcs(S: dict, k: int = 5) -> np.ndarray:
    Xc = kba.cell_mean(S["X"], S)
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
    u, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    Z = u[:, :k] * sv[:k]
    return ((Z - Z.mean(0)) / Z.std(0)).astype(np.float32)


class Occ(torch.nn.Module):
    def __init__(self, n_sp: int, d: int):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(n_sp))
        self.b = torch.nn.Parameter(torch.zeros(d, n_sp))
        self.g = torch.nn.Parameter(torch.full((n_sp,), -1.0))
        self.dl = torch.nn.Parameter(torch.zeros(2))

    def forward(self, Z, obs, morn):
        lpsi = self.a + Z @ self.b
        lp = self.g[None, None, :] + (self.dl[0] * obs + self.dl[1] * morn)[..., None]
        return lpsi, lp


def _nll(model, Z, h, l2):
    lpsi, lp = model(Z, h["obs"], h["morn"])
    y, mask = h["y"], h["mask"][..., None]
    ll_det = (mask * (y * torch.nn.functional.logsigmoid(lp) + (1 - y) * torch.nn.functional.logsigmoid(-lp))).sum(1)
    anyd = (y * mask).sum(1) > 0
    l1 = torch.nn.functional.logsigmoid(lpsi) + ll_det
    l0 = torch.nn.functional.logsigmoid(-lpsi)
    ll = torch.where(anyd, l1, torch.logaddexp(l1, l0))
    return -ll.mean() + l2 * (model.b ** 2).mean()


def fit(Z: np.ndarray, h: dict, steps: int = 3000, l2: float = 1e-2, device: str = "cuda") -> dict:
    T = {k: torch.as_tensor(v, device=device) for k, v in h.items()}
    Zt = torch.as_tensor(Z, device=device)
    model = Occ(h["y"].shape[2], Z.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = _nll(model, Zt, T, l2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        lpsi, lp = model(Zt, T["obs"], T["morn"])
        psi = torch.sigmoid(lpsi)
        logq = (T["mask"][..., None] * torch.nn.functional.logsigmoid(-lp)).sum(1)
        q = torch.exp(logq)
        det = ((T["y"] * T["mask"][..., None]).sum(1) > 0).float()
        pfa = (1 - det) * psi * q / (psi * q + 1 - psi)
        p1 = torch.sigmoid(model.g + model.dl[0] * np.log(2.0))
    return dict(psi=psi.cpu().numpy(), p_list=torch.sigmoid(lp).cpu().numpy(), q=q.cpu().numpy(),
                pfa=pfa.cpu().numpy(), det=det.cpu().numpy(), p_two_obs=p1.cpu().numpy(), nll=float(loss))


def simulate(fitres: dict, h: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    z = rng.random(fitres["psi"].shape) < fitres["psi"]
    y = (rng.random(fitres["p_list"].shape) < fitres["p_list"]) & z[:, None, :] & (h["mask"][..., None] > 0)
    return dict(h, y=y.astype(np.float32)), z


def union_pfa(res: dict) -> np.ndarray:
    det = np.maximum.reduce([res[s]["det"] for s in SEASONS])
    nofa = np.prod([1 - res[s]["pfa"] for s in SEASONS], axis=0)
    return (1 - det) * (1 - nofa)
