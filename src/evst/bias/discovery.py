from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.special import gammaln

AGG = 5


def cell5(df: pd.DataFrame, W: int) -> np.ndarray:
    return ((df.row // AGG) * (W // AGG + 1) + df.col // AGG).to_numpy()


def visit_table(df: pd.DataFrame, W: int) -> pd.DataFrame:
    d = df.assign(c5=cell5(df, W), day=pd.to_datetime(df.eventdate).dt.normalize())
    d["visit"] = d.recordedby.astype(str) + "|" + d.day.astype(str)
    return d


def period_summary(d: pd.DataFrame, lo: int, hi: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = d[(d.year >= lo) & (d.year <= hi)]
    U = p.groupby("c5").visit.nunique().rename("U")
    inc = p.groupby(["c5", "species"]).visit.nunique().rename("Y").reset_index()
    return U, inc


def forward_targets(d: pd.DataFrame, pre: tuple[int, int], post: tuple[int, int]) -> pd.DataFrame:
    U, inc = period_summary(d, *pre)
    m, inc_post = period_summary(d, *post)
    seen = set(zip(inc.c5, inc.species))
    new = inc_post[[(c, s) not in seen for c, s in zip(inc_post.c5, inc_post.species)]].groupby("c5").size()
    stats = inc.groupby("c5").agg(S=("Y", "size"), Q1=("Y", lambda y: (y == 1).sum()), Q2=("Y", lambda y: (y == 2).sum()))
    out = pd.concat([U, stats, m.rename("m"), new.rename("new")], axis=1).fillna(0)
    return out.astype(float)


def shen_chao(U, S, Q1, Q2, m):
    U, S, Q1, Q2, m = (np.asarray(v, float) for v in (U, S, Q1, Q2, m))
    with np.errstate(divide="ignore", invalid="ignore"):
        f0 = np.where(Q2 > 0, (U - 1) / U * Q1 ** 2 / (2 * Q2), (U - 1) / U * Q1 * (Q1 - 1) / 2)
        f0 = np.where(U > 0, np.maximum(f0, 0), 0.0)
        q = np.where(U * f0 + Q1 > 0, Q1 / (U * f0 + Q1), 0.0)
        exp_new = f0 * (1 - (1 - q) ** m)
    return np.nan_to_num(exp_new)


def nb_logpmf(y, mu, phi):
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), 1e-9)
    return (gammaln(y + phi) - gammaln(phi) - gammaln(y + 1) + phi * np.log(phi / (phi + mu))
            + y * np.log(mu / (phi + mu)))


def fit_phi(y, mu):
    grid = np.exp(np.linspace(np.log(0.05), np.log(200), 200))
    ll = [nb_logpmf(y, mu, g).sum() for g in grid]
    return float(grid[int(np.argmax(ll))])


def _lbeta(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def p_undetected(U, a, b):
    return torch.exp(_lbeta(a, b + U) - _lbeta(a, b))


class DiscoveryModel:
    def __init__(self, prior_sd=1.0, device="cuda"):
        self.prior_sd, self.device = prior_sd, device

    def _t(self, x):
        return torch.as_tensor(np.asarray(x, np.float64), device=self.device)

    def _split(self, th):
        k, kd = self.k, self.kd
        return th[:k], th[k:k + kd], th[k + kd:]

    def _nlp(self, th):
        wL, wa, wb = self._split(th)
        Z, Zd, U, S, ci, Y = self.data
        L = torch.exp(Z @ wL)
        a, b = torch.exp(Zd @ wa), torch.exp(Zd @ wb)
        P0 = p_undetected(U, a, b)
        lam = L * (1 - P0)
        ll_S = (S * torch.log(lam + 1e-12) - lam).sum()
        ac, bc, Uc = a[ci], b[ci], U[ci]
        lbb = (self.lgU[ci] - torch.lgamma(Y + 1) - torch.lgamma(Uc - Y + 1)
               + _lbeta(Y + ac, Uc - Y + bc) - _lbeta(ac, bc))
        ll_Y = (lbb - torch.log1p(-P0[ci] + 1e-12)).sum()
        prior = ((wL[1:] ** 2).sum() + (wa[1:] ** 2).sum() + (wb[1:] ** 2).sum()) / (2 * self.prior_sd ** 2)
        return -(ll_S + ll_Y) + prior

    def fit(self, Z, Zd, U, S, inc_cell, inc_Y, steps=600):
        Z, Zd, U, S, Y = (self._t(v) for v in (Z, Zd, U, S, inc_Y))
        ci = torch.as_tensor(np.asarray(inc_cell, np.int64), device=self.device)
        self.data, self.lgU = (Z, Zd, U, S, ci, Y), torch.lgamma(U + 1)
        self.k, self.kd = Z.shape[1], Zd.shape[1]
        th = torch.zeros(self.k + 2 * self.kd, dtype=torch.float64, device=self.device)
        th[0] = float(torch.log(S.mean() * 3 + 1))
        th[self.k], th[self.k + self.kd] = -1.0, 1.0
        th.requires_grad_(True)
        opt = torch.optim.LBFGS([th], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = self._nlp(th)
            loss.backward()
            return loss

        opt.step(closure)
        self.theta = th.detach()
        return self

    def laplace(self, jitter=1e-6):
        H = torch.autograd.functional.hessian(self._nlp, self.theta)
        H = 0.5 * (H + H.T) + jitter * torch.eye(len(self.theta), dtype=H.dtype, device=H.device)
        self.cov = torch.linalg.inv(H)
        return self

    def _mean_new(self, th, Z, Zd, U, m):
        wL, wa, wb = self._split(th)
        L = torch.exp(Z @ wL)
        a, b = torch.exp(Zd @ wa), torch.exp(Zd @ wb)
        return L * (p_undetected(U, a, b) - p_undetected(U + m, a, b))

    @torch.no_grad()
    def expected_new(self, Z, Zd, U, m):
        Z, Zd, U, m = (self._t(v) for v in (Z, Zd, U, m))
        return self._mean_new(self.theta, Z, Zd, U, m).cpu().numpy()

    @torch.no_grad()
    def posterior_new(self, Z, Zd, U, m, draws=200, seed=0):
        Z, Zd, U, m = (self._t(v) for v in (Z, Zd, U, m))
        g = torch.Generator(device=self.device).manual_seed(seed)
        Lc = torch.linalg.cholesky(self.cov)
        eps = torch.randn(draws, len(self.theta), generator=g, dtype=torch.float64, device=self.device)
        ths = self.theta + eps @ Lc.T
        return torch.stack([self._mean_new(th, Z, Zd, U, m) for th in ths]).cpu().numpy()
