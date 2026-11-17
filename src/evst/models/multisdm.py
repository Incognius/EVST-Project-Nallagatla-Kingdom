from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSSES = ("an_ssdl", "an_slds", "an_full", "maxent_random", "maxent_tgb", "occ_effort")


class Net(nn.Module):
    def __init__(self, d_in, n_sp, hidden=256, depth=3, drop=0.2):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(drop)]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_sp)
        self.det = nn.Parameter(torch.full((n_sp,), -2.0))
        self.balanced = True

    def forward(self, x):
        return self.head(self.body(x))


def _balanced_bce(logit, y, mask=None):
    if mask is None:
        mask = torch.ones_like(y)
    pos, neg = y * mask, (1 - y) * mask
    lp = -(F.logsigmoid(logit) * pos).sum(0) / pos.sum(0).clamp(min=1)
    ln = -(F.logsigmoid(-logit) * neg).sum(0) / neg.sum(0).clamp(min=1)
    return (lp + ln).mean()


def _maxent(f_pos_rows, y, f_bg):
    n = y.sum(0).clamp(min=1)
    return (-(f_pos_rows * y).sum(0) / n + torch.logsumexp(f_bg, 0) - np.log(f_bg.shape[0])).mean()


def po_loss(net, loss, Xs, Ys, Us, Xland, n_bg, gen):
    f_s = net(Xs)
    if loss == "an_slds":
        return _balanced_bce(f_s, Ys)
    if loss == "occ_effort":
        det = 1 - torch.exp(Us[:, None] * F.logsigmoid(-net.det)[None, :])
        p = torch.sigmoid(f_s) * det
        p = p.clamp(1e-6, 1 - 1e-6)
        w1 = Ys.sum(0).clamp(min=1) if net.balanced else torch.full_like(Ys[0], float(Ys.shape[0]))
        w0 = (1 - Ys).sum(0).clamp(min=1) if net.balanced else w1
        return (-(Ys * torch.log(p)).sum(0) / w1 - ((1 - Ys) * torch.log(1 - p)).sum(0) / w0).mean()
    if loss == "maxent_tgb":
        return _maxent(f_s, Ys, f_s)
    idx = torch.randint(0, Xland.shape[0], (n_bg,), device=Xland.device, generator=gen)
    f_r = net(Xland[idx])
    if loss == "maxent_random":
        return _maxent(f_s, Ys, f_r)
    yr = torch.zeros(n_bg, Ys.shape[1], device=Ys.device)
    if loss == "an_ssdl":
        logit = torch.cat([f_s, f_r])
        y = torch.cat([Ys, yr])
        m = torch.cat([Ys, torch.ones_like(yr)])
        return _balanced_bce(logit, y, m)
    return _balanced_bce(torch.cat([f_s, f_r]), torch.cat([Ys, yr]))


def score(net, loss, X, chunk=65536):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            f = net(X[i:i + chunk])
            out.append((torch.exp(f - f.max()) if loss.startswith("maxent") else torch.sigmoid(f)).cpu())
    net.train()
    return torch.cat(out).numpy()


def fit(loss, Xland, tg_li, Ytg, Utg, steps=1500, seed=0, lr=1e-3, wd=1e-4, n_bg=8192, hidden=256, drop=0.2,
        pa=None, pa_weight=1.0, po_weight=1.0, init=None, balanced=True, device="cuda"):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    net = Net(Xland.shape[1], Ytg.shape[1], hidden=hidden, drop=drop).to(device)
    net.balanced = balanced
    if init is not None:
        net.load_state_dict(init)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xs, Ys = Xland[tg_li], Ytg
    for _ in range(steps):
        L = po_weight * po_loss(net, loss, Xs, Ys, Utg, Xland, n_bg, gen) if po_weight > 0 else 0.0
        if pa is not None:
            pix, seg, ncell, Ypa = pa
            pp = torch.sigmoid(net(Xland[pix]))
            cp = torch.zeros(ncell, Ypa.shape[1], device=device).index_add_(0, seg, pp)
            cp = (cp / torch.bincount(seg, minlength=ncell)[:, None]).clamp(1e-5, 1 - 1e-5)
            L = L + pa_weight * F.binary_cross_entropy(cp, Ypa)
        opt.zero_grad()
        L.backward()
        opt.step()
    return net


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def fit_adv(loss, Xland, Aland, tg_li, Ytg, Utg, adv=1.0, steps=1500, seed=0, lr=1e-3, wd=1e-4, n_bg=8192,
            hidden=256, drop=0.2, device="cuda"):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    net = Net(Xland.shape[1], Ytg.shape[1], hidden=hidden, drop=drop).to(device)
    head = nn.Sequential(nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, Aland.shape[1])).to(device)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
    Xs, Ys = Xland[tg_li], Ytg
    for _ in range(steps):
        L = po_loss(net, loss, Xs, Ys, Utg, Xland, n_bg, gen)
        if adv > 0:
            idx = torch.cat([tg_li, torch.randint(0, Xland.shape[0], (n_bg,), device=device, generator=gen)])
            h = _GRL.apply(net.body(Xland[idx]), adv)
            L = L + F.mse_loss(head(h), Aland[idx])
        opt.zero_grad()
        L.backward()
        opt.step()
    return net


def embed(net, X, chunk=65536):
    net.eval()
    with torch.no_grad():
        out = torch.cat([net.body(X[i:i + chunk]).cpu() for i in range(0, X.shape[0], chunk)])
    net.train()
    return out.numpy()
