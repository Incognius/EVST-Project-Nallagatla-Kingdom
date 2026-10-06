from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VirtualSpecies:
    suit: np.ndarray
    prevalence: float
    params: dict


def make_species(E: np.ndarray, rng: np.random.Generator, n_vars: int = 3,
                 prevalence: float | None = None) -> VirtualSpecies:
    Z = (E - E.mean(0)) / (E.std(0) + 1e-9)
    idx = rng.choice(Z.shape[1], n_vars, replace=False)
    mu = rng.uniform(-1.5, 1.5, n_vars)
    sd = rng.uniform(0.5, 1.5, n_vars)
    lin = -0.5 * (((Z[:, idx] - mu) / sd) ** 2).sum(1)
    rho = rng.uniform(-0.5, 0.5)
    lin += rho * (Z[:, idx[0]] - mu[0]) * (Z[:, idx[1]] - mu[1])
    prevalence = prevalence or rng.uniform(0.05, 0.35)
    lo, hi = -50.0, 50.0
    for _ in range(60):
        b = (lo + hi) / 2
        m = (1 / (1 + np.exp(-(lin - lin.mean()) * 1.5 - b))).mean()
        lo, hi = (b, hi) if m < prevalence else (lo, b)
    suit = 1 / (1 + np.exp(-(lin - lin.mean()) * 1.5 - b))
    return VirtualSpecies(suit, prevalence, {"vars": idx.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "rho": rho})


def sample_presences(sp: VirtualSpecies, effort: np.ndarray, strength: float, n: int,
                     rng: np.random.Generator) -> np.ndarray:
    occupied = rng.random(len(sp.suit)) < sp.suit
    w = np.where(occupied, np.power(np.clip(effort, 1e-12, None), strength), 0.0)
    w /= w.sum()
    return rng.choice(len(w), size=min(n, int((w > 0).sum())), replace=False, p=w)
