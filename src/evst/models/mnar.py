from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

KINDS = ("mf", "env_tg", "env_mf_tg", "occ_unit", "sar_pu", "expo_em", "expo_em_noacc")


def units(S: dict, step: float = 1 / 16) -> dict:
    n_ev = len(S["ev"])
    u = np.full(len(S["land"]), -1)
    u[S["pix_li"]] = S["pix_k"]
    rest = u < 0
    g = np.floor(S["lonlat"][rest] / step).astype(int)
    _, inv = np.unique(g[:, 0] * 100000 + g[:, 1], return_inverse=True)
    u[rest] = n_ev + inv
    n = u.max() + 1
    X = S["X"]
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)
    A = S["A"]
    Az = (A - A.mean(0)) / (A.std(0) + 1e-6)
    cnt = np.bincount(u, minlength=n).astype(float)
    E = np.zeros((n, Xz.shape[1]))
    np.add.at(E, u, Xz)
    Ac = np.zeros((n, Az.shape[1]))
    np.add.at(Ac, u, Az)
    b = S["birds"]
    bu = u[b.li.to_numpy()]
    sidx = pd.Series(np.arange(len(S["cand"])), index=S["cand"])
    bc = b.assign(unit=bu)[b.species.isin(S["cand"]).to_numpy()]
    R = np.zeros((n, len(S["cand"])), np.float32)
    R[bc.unit.to_numpy(), sidx[bc.species].to_numpy()] = 1
    U = b.assign(unit=bu).groupby("unit").visit.nunique().reindex(np.arange(n), fill_value=0).to_numpy(np.float32)
    return dict(unit=u, n=n, n_ev=n_ev, E=(E / cnt[:, None]).astype(np.float32), Acc=(Ac / cnt[:, None]).astype(np.float32),
                R=R, U=U, npix=cnt)


def knn(R: np.ndarray, U: np.ndarray, k: int = 20) -> np.ndarray:
    vis = np.flatnonzero(U > 0)
    Rv = R[vis]
    nrm = np.linalg.norm(Rv, axis=1, keepdims=True) + 1e-9
    sim = (Rv / nrm) @ (Rv / nrm).T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argsort(-sim, axis=1)[:, :k]
    w = np.clip(np.take_along_axis(sim, idx, 1), 0, None) + 1e-6
    out = np.tile(R[vis].mean(0), (len(R), 1))
    out[vis] = (w[..., None] * Rv[idx]).sum(1) / w.sum(1, keepdims=True)
    return out


class Model(torch.nn.Module):
    def __init__(self, n_units, n_sp, d_env, d_acc, env=True, latent=0, hidden=128, k=32, drop=0.2):
        super().__init__()
        self.env = torch.nn.Sequential(torch.nn.Linear(d_env, hidden), torch.nn.SiLU(), torch.nn.Dropout(drop),
                                       torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Dropout(drop),
                                       torch.nn.Linear(hidden, k)) if env else None
        self.v = torch.nn.Parameter(torch.randn(k, n_sp) * 0.05)
        self.b = torch.nn.Parameter(torch.full((n_sp,), -2.0))
        self.latent = latent
        if latent:
            self.u = torch.nn.Parameter(torch.randn(n_units, latent) * 0.01)
            self.w = torch.nn.Parameter(torch.randn(latent, n_sp) * 0.01)
        self.det = torch.nn.Parameter(torch.full((n_sp,), -2.0))
        self.ea = torch.nn.Parameter(torch.full((n_sp,), -2.0))
        self.eg = torch.nn.Parameter(torch.zeros(d_acc))

    def psi_logit(self, E):
        z = self.b.expand(E.shape[0], -1)
        if self.env is not None:
            z = z + self.env(E) @ self.v
        if self.latent:
            z = z + self.u @ self.w
        return z

    def exposure_log(self, U, Acc, use_acc=True):
        lp = self.ea[None, :] + ((Acc @ self.eg)[:, None] if use_acc else 0.0)
        lq = U[:, None] * F.logsigmoid(-lp)
        return _log1mexp(lq.clamp(max=-1e-6)), lq


def _log1mexp(x):
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def fit(kind: str, T: dict, seed: int = 0, steps: int = 2000, lr: float = 3e-3, wd: float = 1e-4, l2_lat: float = 1e-3,
        latent: int = 8, p_fixed=None, device: str = "cuda"):
    torch.manual_seed(seed)
    n, m = T["R"].shape
    env = kind != "mf"
    lat = latent if kind in ("mf", "env_mf_tg", "sar_pu", "expo_em", "expo_em_noacc") else 0
    net = Model(n, m, T["E"].shape[1], T["Acc"].shape[1], env=env, latent=lat).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    R, U, E, Acc = T["R"], T["U"], T["E"], T["Acc"]
    vis = U > 0
    if kind == "sar_pu":
        e = (1 - (1 - p_fixed[None, :]) ** U[:, None]).clamp(0.05, 1.0)
    for it in range(steps):
        net.train()
        opt.zero_grad()
        z = net.psi_logit(E)
        if kind in ("mf", "env_tg", "env_mf_tg"):
            loss = F.binary_cross_entropy_with_logits(z[vis], R[vis])
        elif kind == "occ_unit":
            lq = U[:, None] * F.logsigmoid(-net.det)[None, :]
            lp1 = F.logsigmoid(z) + _log1mexp(lq.clamp(max=-1e-6))
            lp0 = torch.log1p(-torch.exp(lp1).clamp(max=1 - 1e-6))
            loss = -(R * lp1 + (1 - R) * lp0).mean()
        elif kind == "sar_pu":
            lpos = F.binary_cross_entropy_with_logits(z, torch.ones_like(z), reduction="none")
            lneg = F.binary_cross_entropy_with_logits(z, torch.zeros_like(z), reduction="none")
            loss = (R * (lpos / e + (1 - 1 / e) * lneg) + (1 - R) * lneg).mean()
        else:
            lmu, lnmu = net.exposure_log(U, Acc, use_acc=kind == "expo_em")
            with torch.no_grad():
                ps, mu = torch.sigmoid(z), torch.exp(lmu)
                den = (1 - ps * mu).clamp(min=1e-6)
                tz = torch.where(R > 0, torch.ones_like(ps), ps * (1 - mu) / den)
                ta = torch.where(R > 0, torch.ones_like(ps), mu * (1 - ps) / den)
            loss = F.binary_cross_entropy_with_logits(z, tz) - (ta * lmu + (1 - ta) * lnmu).mean()
        if lat:
            loss = loss + l2_lat * ((net.u ** 2).sum(1).mean() + (net.w ** 2).sum(0).mean())
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        psi = torch.sigmoid(net.psi_logit(E)).cpu().numpy()
        extra = {"det": torch.sigmoid(net.det).cpu().numpy(), "eg": net.eg.detach().cpu().numpy()}
    return psi, extra


def tensors(Ut: dict, device: str = "cuda") -> dict:
    return {k: torch.as_tensor(Ut[k], device=device) for k in ("R", "U", "E", "Acc")}
