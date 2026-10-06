from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

A_Q, B_Q = 1.0, 0.01
A_B, B_B = 1.0, 0.001


def fit_map(counts: np.ndarray, D: np.ndarray, names: list[str]) -> dict:
    X = np.asarray(D, float)
    y = np.asarray(counts, float)
    J = X.shape[1]

    def nlp(theta):
        lq, lw, lb = theta[0], theta[1:1 + J], theta[-1]
        q, w, b = np.exp(lq), np.exp(lw), np.exp(lb)
        eta = lq - X @ w
        ll = (y * eta - np.exp(eta)).sum()
        lp = (A_Q - 1) * lq - B_Q * q + lq
        lp += (J * np.log(b) - b * w.sum()) + lw.sum()
        lp += (A_B - 1) * lb - B_B * b + lb
        return -(ll + lp)

    x0 = np.r_[np.log(y.mean() + 1e-9), np.full(J, np.log(1e-3)), 0.0]
    r = minimize(nlp, x0, method="L-BFGS-B")
    w = np.exp(r.x[1:1 + J])
    return {"q": float(np.exp(r.x[0])), "w": dict(zip(names, w)), "b": float(np.exp(r.x[-1])),
            "converged": bool(r.success)}


def fit_nuts(counts: np.ndarray, D: np.ndarray, names: list[str], draws: int = 1000, tune: int = 1000,
             seed: int = 0) -> dict:
    import pymc as pm
    X = np.asarray(D, float)
    y = np.asarray(counts, int)
    with pm.Model():
        q = pm.Gamma("q", alpha=A_Q, beta=B_Q)
        b = pm.Gamma("b", alpha=A_B, beta=B_B)
        w = pm.Gamma("w", alpha=1.0, beta=b, shape=X.shape[1])
        pm.Poisson("y", mu=q * pm.math.exp(-pm.math.dot(X, w)), observed=y)
        tr = pm.sample(draws, tune=tune, chains=2, cores=1, random_seed=seed, progressbar=False,
                       target_accept=0.9)
    ws = tr.posterior["w"].values.reshape(-1, X.shape[1])
    return {"q_mean": float(tr.posterior["q"].values.mean()),
            "w_mean": dict(zip(names, ws.mean(0))), "w_q025": dict(zip(names, np.quantile(ws, 0.025, 0))),
            "w_q975": dict(zip(names, np.quantile(ws, 0.975, 0)))}


def predict_rate(fit: dict, D: np.ndarray, names: list[str]) -> np.ndarray:
    w = np.array([fit["w"][n] if "w" in fit else fit["w_mean"][n] for n in names])
    q = fit.get("q", fit.get("q_mean"))
    return q * np.exp(-np.asarray(D, float) @ w)
