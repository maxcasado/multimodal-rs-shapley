"""Small statistics used by the local analyses: weighted correlations, Moran's I (queen
contiguity on the lon/lat grid) with a vectorised permutation test, one-way fixed-effect R²."""
import numpy as np
from scipy.stats import rankdata


def wpearson(x, y, w=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 3:
        return np.nan
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx, vy = np.average((x - mx) ** 2, weights=w), np.average((y - my) ** 2, weights=w)
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def wspearman(x, y, w=None):
    """Weighted Pearson on the (unweighted, average-tie) ranks."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if w is not None:
        w = np.asarray(w, float)
        ok &= np.isfinite(w) & (w > 0)
        w = w[ok]
    return wpearson(rankdata(x[ok]), rankdata(y[ok]), w)


def _queen_pairs(grid):
    """Ordered neighbour pairs (i, j) among the finite cells of ``grid``, queen
    contiguity on the (lon, lat) index grid, no wrap-around."""
    n_lon, n_lat = grid.shape
    flat = grid.ravel()
    idx = np.where(np.isfinite(flat))[0]
    pos = -np.ones(flat.size, int)
    pos[idx] = np.arange(len(idx))
    r, c = idx // n_lat, idx % n_lat
    I, J = [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            ok = (nr >= 0) & (nr < n_lon) & (nc >= 0) & (nc < n_lat)
            nb = np.where(ok, nr * n_lat + nc, 0)
            ok &= pos[nb] >= 0
            I.append(np.arange(len(idx))[ok])
            J.append(pos[nb[ok]])
    return flat[idx], np.concatenate(I), np.concatenate(J)


def moran(grid, permutations=199, seed=0):
    """Moran's I of a (n_lon, n_lat) grid (NaN = no cell): binary queen weights between
    finite cells, no wrap-around. Two-sided permutation p-value (|I_perm| >= |I|) and
    z-score (I - E[I]) / std(I_perm), E[I] = -1/(n-1)."""
    x, I, J = _queen_pairs(grid)
    n = len(x)
    xd = x - x.mean()
    if n < 4 or len(I) == 0 or (xd ** 2).sum() == 0:
        return {"I": np.nan, "p_value": np.nan, "z_score": np.nan, "n_cells": n}
    W = float(len(I))
    I_obs = (n / W) * (xd[I] * xd[J]).sum() / (xd ** 2).sum()
    rng = np.random.default_rng(seed)
    X = np.stack([rng.permutation(x) for _ in range(int(permutations))])
    Xd = X - X.mean(1, keepdims=True)
    sim = (n / W) * (Xd[:, I] * Xd[:, J]).sum(1) / (Xd ** 2).sum(1)
    p = (np.sum(np.abs(sim) >= abs(I_obs)) + 1) / (permutations + 1)
    return {"I": float(I_obs), "p_value": float(p),
            "z_score": float((I_obs + 1.0 / (n - 1)) / sim.std()), "n_cells": n}


def fixed_effect_r2(y, groups):
    """R² and adjusted R² of y ~ group fixed effects (one-way ANOVA)."""
    y = np.asarray(y, float)
    g = np.asarray(groups)
    ok = np.isfinite(y)
    y, g = y[ok], g[ok]
    _, inv = np.unique(g, return_inverse=True)
    k = inv.max() + 1
    means = np.bincount(inv, weights=y) / np.bincount(inv)
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - means[inv]) ** 2).sum()
    n = len(y)
    r2 = 1 - ss_res / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else np.nan
    return float(r2), float(adj), int(k), int(n)
