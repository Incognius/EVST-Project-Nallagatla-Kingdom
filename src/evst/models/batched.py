from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GLinear(nn.Module):
    def __init__(self, S, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.randn(S, i, o) * (2 / i) ** 0.5)
        self.b = nn.Parameter(torch.zeros(S, 1, o))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)


class BatchedMLP(nn.Module):
    def __init__(self, S, c_in, h=256, depth=3):
        super().__init__()
        dims = [c_in] + [h] * depth
        self.layers = nn.ModuleList([GLinear(S, a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = GLinear(S, h, 1)

    def forward(self, patches):
        x = patches[:, :, :, patches.shape[3] // 2, patches.shape[4] // 2]
        for l in self.layers:
            x = torch.relu(l(x))
        return self.out(x).squeeze(-1)


class BatchedCNN(nn.Module):
    def __init__(self, S, c_in, width=32):
        super().__init__()
        self.S = S
        g = lambda i, o: nn.Conv2d(S * i, S * o, 3, padding=1, groups=S)
        self.f = nn.Sequential(g(c_in, width), nn.BatchNorm2d(S * width), nn.ReLU(),
                               g(width, width), nn.BatchNorm2d(S * width), nn.ReLU(), nn.MaxPool2d(2),
                               g(width, 2 * width), nn.BatchNorm2d(S * 2 * width), nn.ReLU(),
                               nn.AdaptiveAvgPool2d(1))
        self.h1 = GLinear(S, 2 * width + c_in, 128)
        self.h2 = GLinear(S, 128, 1)

    def forward(self, patches):
        S, B, C, P, _ = patches.shape
        x = patches.permute(1, 0, 2, 3, 4).reshape(B, S * C, P, P)
        f = self.f(x).view(B, S, -1).permute(1, 0, 2)
        centre = patches[:, :, :, P // 2, P // 2]
        z = torch.relu(self.h1(torch.cat([f, centre], -1)))
        return self.h2(z).squeeze(-1)


class GPUStack:

    def __init__(self, stack: np.ndarray, patch: int = 15, device: str = "cuda"):
        self.r, self.p = patch // 2, patch
        S = stack.astype(np.float32)
        mu = np.nanmean(S, axis=(1, 2), keepdims=True)
        sd = np.nanstd(S, axis=(1, 2), keepdims=True) + 1e-6
        self.mu, self.sd = mu.ravel(), sd.ravel()
        S = np.nan_to_num((S - mu) / sd, nan=0.0)
        self.H, self.W = S.shape[1:]
        self.S = torch.tensor(np.pad(S, ((0, 0), (self.r, self.r), (self.r, self.r))), device=device)
        o = torch.arange(patch, device=device)
        self.dr, self.dc = o.view(1, 1, patch, 1), o.view(1, 1, 1, patch)

    def patches(self, cells: torch.Tensor, override: dict | None = None, centre_only: bool = False) -> torch.Tensor:
        if centre_only:
            r = (cells // self.W + self.r).unsqueeze(-1).unsqueeze(-1)
            c = (cells % self.W + self.r).unsqueeze(-1).unsqueeze(-1)
        else:
            r = (cells // self.W).unsqueeze(-1).unsqueeze(-1) + self.dr
            c = (cells % self.W).unsqueeze(-1).unsqueeze(-1) + self.dc
        x = self.S[:, r, c].permute(1, 2, 0, 3, 4)
        if override:
            x = x.clone()
            for ch, v in override.items():
                x[:, :, ch] = v
        return x


def fit_predict(kind: str, stack: GPUStack, pres: list[np.ndarray], bg: list[np.ndarray],
                eval_cells: np.ndarray, steps: int = 1200, bs: int = 128, lr: float = 2e-3,
                seed: int = 0, override: dict | None = None, channels: list[int] | None = None,
                extra_cells: np.ndarray | None = None):
    torch.manual_seed(seed)
    dev = stack.S.device
    S = len(pres)
    ch = torch.tensor(channels if channels is not None else list(range(stack.S.shape[0])), device=dev)
    C = len(ch)
    net = (BatchedMLP(S, C) if kind == "mlp" else BatchedCNN(S, C)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)

    def pad(lst):
        L = max(len(a) for a in lst)
        out = np.zeros((len(lst), L), np.int64)
        for i, a in enumerate(lst):
            out[i, :len(a)] = a
        return torch.tensor(out, device=dev), torch.tensor([len(a) for a in lst], device=dev)

    P, nP = pad(pres)
    Bg, nB = pad(bg)
    half = bs // 2
    y = torch.cat([torch.ones(S, half, device=dev), torch.zeros(S, half, device=dev)], 1)
    net.train()
    for _ in range(steps):
        ip = (torch.rand(S, half, device=dev) * nP[:, None]).long()
        ib = (torch.rand(S, half, device=dev) * nB[:, None]).long()
        cells = torch.cat([P.gather(1, ip), Bg.gather(1, ib)], 1)
        x = stack.patches(cells, centre_only=kind == "mlp")[:, :, ch]
        if kind == "cnn" and torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(net(x), y, reduction="mean") * S
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    ov = None
    if override:
        pos = {int(c): i for i, c in enumerate(ch.tolist())}
        ov = {pos[c]: v for c, v in override.items() if c in pos}

    def run(cells_np, ov_):
        out, ev = [], torch.tensor(cells_np, device=dev)
        with torch.no_grad():
            chunk = max(256, 32768 // S) if kind == "cnn" else 8192
            for i in range(0, len(ev), chunk):
                cells = ev[i:i + chunk].unsqueeze(0).expand(S, -1)
                x = stack.patches(cells, centre_only=kind == "mlp")[:, :, ch]
                if ov_:
                    x = x.clone()
                    for k, v in ov_.items():
                        x[:, :, k] = v
                out.append(torch.sigmoid(net(x)).cpu())
        return torch.cat(out, 1).numpy()

    main = run(eval_cells, ov)
    return main if extra_cells is None else (main, run(extra_cells, None))
