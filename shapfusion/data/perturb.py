"""
The two S2 degradations of the paper, applied to every item of a dataset (train AND test):
the scenario is a sensor that is degraded for good, and the model is retrained at every level.

GaussianNoise (experiment ``noise``)
    N(0, sigma^2) added to the z-scored S2_S2VI array of the item. The draw is a pure
    function of (sigma, fold, position of the item in the dataset):
        seed = seed_offset + 7919 * fold + round(1e4 * sigma) + index
    so it is identical across processes and DataLoader workers. sigma = 0 is a no-op.

CloudGap (experiment ``cloud_gap``)
    k of the T = 12 acquisition dates of S2_S2VI are dropped and refilled by linear
    interpolation over the surviving dates (edge-clamped outside the observed range); at
    k = T the series is set to 0, the per-band mean in z-space. The drop order is a
    permutation of the T dates drawn once per sample from seed_offset + crc32(identifier),
    and level k drops its first k entries: masks are nested across k, identical in every
    fold, in train and test mode. Interpolating after the per-band z-score is exact
    (an affine map constant in time commutes with linear interpolation). k = 0 is a no-op.
"""
from __future__ import annotations

import zlib

import numpy as np


class GaussianNoise:
    def __init__(self, target: str, sigma: float, fold: int, seed_offset: int):
        self.target, self.sigma = target, float(sigma)
        self.seed_base = int(seed_offset) + int(fold) * 7919 + int(round(self.sigma * 10_000))

    def __call__(self, item: dict, index: int) -> dict:
        if self.sigma > 0.0 and self.target in item["views"]:
            arr = item["views"][self.target]
            rng = np.random.default_rng(self.seed_base + index)
            noise = rng.normal(loc=0.0, scale=self.sigma, size=arr.shape).astype(arr.dtype)
            item["views"][self.target] = arr + noise
        return item


def interpolate_missing(arr: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    """(T, C) series rebuilt from the dates ``keep_idx`` (sorted) by linear interpolation;
    observed dates are kept bit-exact, an empty ``keep_idx`` gives zeros."""
    T = arr.shape[0]
    if keep_idx.size == 0:
        return np.zeros_like(arr)
    if keep_idx.size == T:
        return arr
    t = np.arange(T)
    j = np.searchsorted(keep_idx, t, side="left")
    j_hi = np.clip(j, 0, keep_idx.size - 1)
    j_lo = np.clip(j - 1, 0, keep_idx.size - 1)
    t_lo = keep_idx[j_lo].astype(np.float64)
    t_hi = keep_idx[j_hi].astype(np.float64)
    span = t_hi - t_lo
    w = np.where(span > 0, (t - t_lo) / np.where(span > 0, span, 1.0), 0.0)
    lo, hi = arr[keep_idx[j_lo]], arr[keep_idx[j_hi]]
    out = lo + w[:, None] * (hi - lo)
    out[keep_idx] = arr[keep_idx]
    return out.astype(arr.dtype, copy=False)


def keep_indices(identifier, n_missing: int, T: int, seed_offset: int) -> np.ndarray:
    """Sorted indices of the observed dates of one sample (nested across n_missing)."""
    seed = int(seed_offset) + zlib.crc32(str(identifier).encode("utf-8"))
    order = np.random.default_rng(seed).permutation(T)
    return np.sort(order[n_missing:])


class CloudGap:
    def __init__(self, target: str, k: int, seq_len: int, seed_offset: int):
        self.target, self.k, self.seq_len, self.seed_offset = target, int(k), int(seq_len), int(seed_offset)

    def __call__(self, item: dict, index: int) -> dict:
        if self.k > 0 and self.target in item["views"]:
            arr = item["views"][self.target]
            keep = keep_indices(item["identifier"], self.k, arr.shape[0], self.seed_offset)
            item["views"][self.target] = interpolate_missing(arr, keep)
        return item


def for_job(cfg: dict, experiment: str, level, fold: int):
    """The perturbation of a job, or None."""
    spec = cfg["experiments"][experiment]
    if experiment == "noise":
        return GaussianNoise(spec["target"], level, fold, spec["seed_offset"])
    if experiment == "cloud_gap":
        return CloudGap(spec["target"], level, spec["seq_len"], spec["seed_offset"])
    return None
