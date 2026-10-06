from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


def _balance_weights(y: np.ndarray) -> np.ndarray:
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return np.where(y == 1, 1.0, n1 / n0)


class GLM:
    name = "GLM"

    def __init__(self, C: float = 1.0):
        self.m = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(),
                               LogisticRegression(C=C, max_iter=2000))

    def fit(self, X, y):
        self.m.fit(X, y, logisticregression__sample_weight=_balance_weights(y))
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


class MaxEnt:
    name = "MaxEnt"

    def __init__(self, beta: float = 1.0):
        import elapid
        self.m = elapid.MaxentModel(feature_types=["linear", "quadratic", "hinge", "product"],
                                    beta_multiplier=beta, transform="cloglog", n_cpus=1)

    def fit(self, X, y):
        self.m.fit(np.asarray(X, dtype=float), np.asarray(y))
        return self

    def predict(self, X):
        return self.m.predict(np.asarray(X, dtype=float))


class BRT:
    name = "BRT"

    def __init__(self, n_estimators: int = 600, lr: float = 0.02, seed: int = 0):
        import lightgbm as lgb
        self.m = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=lr, num_leaves=8,
                                    min_child_samples=5, subsample=0.75, subsample_freq=1,
                                    colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=1)

    def fit(self, X, y):
        self.m.fit(X, y, sample_weight=_balance_weights(y))
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


class RFDown:
    name = "RF"

    def __init__(self, n_trees: int = 300, seed: int = 0):
        self.n_trees, self.seed = n_trees, seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        p, b = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        mtry = max(1, int(np.sqrt(X.shape[1])))
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.concatenate([rng.choice(p, len(p)), rng.choice(b, len(p))])
            t = DecisionTreeClassifier(max_features=mtry, random_state=int(rng.integers(1 << 31)))
            self.trees.append(t.fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.mean([t.predict_proba(X)[:, 1] for t in self.trees], axis=0)


class MLP:
    name = "MLP"

    def __init__(self, hidden: int = 256, depth: int = 3, steps: int = 1500, bs: int = 256, lr: float = 1e-3,
                 wd: float = 1e-4, seed: int = 0, device: str | None = None):
        import torch
        self.h, self.d, self.steps, self.bs, self.lr, self.wd, self.seed = hidden, depth, steps, bs, lr, wd, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        self.sc = StandardScaler().fit(X)
        Xt = torch.tensor(self.sc.transform(X), dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)
        p, b = torch.nonzero(yt == 1).squeeze(1), torch.nonzero(yt == 0).squeeze(1)
        layers, d_in = [], Xt.shape[1]
        for _ in range(self.d):
            layers += [nn.Linear(d_in, self.h), nn.ReLU()]
            d_in = self.h
        self.net = nn.Sequential(*layers, nn.Linear(d_in, 1)).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.steps)
        for _ in range(self.steps):
            j = torch.cat([p[torch.randint(len(p), (self.bs // 2,), device=self.device)],
                           b[torch.randint(len(b), (self.bs // 2,), device=self.device)]])
            loss = nn.functional.binary_cross_entropy_with_logits(self.net(Xt[j]).squeeze(1), yt[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        return self

    def predict(self, X):
        import torch
        with torch.no_grad():
            Xt = torch.tensor(self.sc.transform(X), dtype=torch.float32, device=self.device)
            return torch.sigmoid(self.net(Xt).squeeze(1)).cpu().numpy()


MODELS = {"GLM": GLM, "MaxEnt": MaxEnt, "BRT": BRT, "RF": RFDown, "MLP": MLP}
