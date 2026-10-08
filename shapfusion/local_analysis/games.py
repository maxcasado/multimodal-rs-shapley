"""
Per-sample games built from the cached local v-dicts, and their Shapley values /
interaction indices (shapley/core.py, array-valued games: one game per sample).

Value functions (``vf``), all with v(∅) = cfg["v_empty"] (0.5, uniform prior):
    prob_pred     v(S) = P(ŷ_full | x_S)                 (the local maps of the paper)
    logodds_pred  v(S) = log-odds of ŷ_full given x_S,   v(∅) = b = logit(v_empty)
    prob_true     v(S) = P(y_true | x_S)
ŷ_full = 1[P(class 1 | all four sensors) > 0.5].
"""
import itertools
import math

import numpy as np
import pandas as pd

from shapfusion.shapley import core as sh
from . import vcache

VALUE_FUNCTIONS = {
    "prob_pred":    "v(S) = P(ŷ_full | x_S), v(∅) = {v0:g}",
    "logodds_pred": "v(S) = log-odds(ŷ_full | x_S), v(∅) = logit({v0:g}) = {b:g}",
    "prob_true":    "v(S) = P(y_true | x_S), v(∅) = {v0:g}",
}


def vf_label(cfg, vf):
    v0 = float(cfg["v_empty"])
    return VALUE_FUNCTIONS[vf].format(v0=v0, b=_logit(v0))


def _logit(p):
    return math.log(p) - math.log1p(-p)


def short(v):
    return "S2" if v == "S2_S2VI" else v


def pairs(views):
    return list(itertools.combinations(views, 2))


def pair_name(a, b):
    return f"{short(a)}x{short(b)}"


def full_mask(views):
    return (1 << len(views)) - 1


def yhat_full(cache, views):
    return (cache["p1"][full_mask(views)] > 0.5).astype(np.int64)


def value_table(cache, views, vf, v_empty=0.5, logit_key="logit"):
    """(16, N) table of v(S, x) under value function ``vf`` (row m = bitmask)."""
    p1 = cache["p1"]
    if vf == "prob_pred":
        sign = np.where(yhat_full(cache, views) == 1, 1.0, -1.0)
        T = np.where(sign > 0, p1, 1.0 - p1)
        T[0] = v_empty
    elif vf == "prob_true":
        y = cache["y_true"]
        T = np.where(y == 1, p1, 1.0 - p1)
        T[0] = v_empty
    elif vf in ("logodds_pred", "logodds_pred64"):
        sign = np.where(yhat_full(cache, views) == 1, 1.0, -1.0)
        T = sign * cache[logit_key]
        T[0] = _logit(v_empty)
    else:
        raise ValueError(vf)
    return T.astype(np.float64)


def as_vdict(T, views):
    return {vcache.members(m, views): T[m] for m in range(T.shape[0])}


def phi(T, views):
    """(N, n_views) Shapley values."""
    d = sh.shapley_values(views, as_vdict(T, views))
    return np.stack([d[v] for v in views], axis=1)


def sii(T, views):
    """(N, n_pairs) order-2 interaction index."""
    d = sh.shapley_interactions(views, as_vdict(T, views))
    return np.stack([d[p] for p in pairs(views)], axis=1)


def sii_terms(T, views):
    """{pair: (N, 3)} weighted context-size terms [1/3 Δ(∅), 1/6 ΣΔ({k}), 1/3 Δ({k,l})]."""
    d = sh.shapley_interaction_terms(views, as_vdict(T, views))
    return {p: np.stack([d[p][s] for s in range(len(views) - 1)], axis=1) for p in pairs(views)}


def sample_frame(cfg, model, vf="prob_pred", with_terms=False, with_regions=True):
    """One row per test sample (all folds stacked): metadata + phi_<view> + I_<pair>
    (+ T<s>_<pair> terms) under ``vf``."""
    views = cfg["views"]
    rows = []
    for fold in cfg["folds"]:
        c = vcache.load(cfg, model, fold)
        T = value_table(c, views, vf, cfg["v_empty"])
        P = phi(T, views)
        I = sii(T, views)
        d = {"fold": fold, "sample_idx": c["sample_idx"], "identifier": c["identifier"],
             "lon": c["lon"], "lat": c["lat"], "y_true": c["y_true"],
             "yhat": yhat_full(c, views), "v_full": T[full_mask(views)], "v_empty": T[0]}
        d["correct"] = (d["yhat"] == d["y_true"]).astype(np.int64)
        for k, v in enumerate(views):
            d[f"phi_{short(v)}"] = P[:, k]
        for k, (a, b) in enumerate(pairs(views)):
            d[f"I_{pair_name(a, b)}"] = I[:, k]
        if with_terms:
            for (a, b), M in sii_terms(T, views).items():
                for s in range(M.shape[1]):
                    d[f"T{s}_{pair_name(a, b)}"] = M[:, s]
        rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    df["subdataset"] = df.identifier.str.split("_", n=1).str[1]
    if with_regions:
        df = df.merge(regions(cfg), on=["fold", "sample_idx"], how="left", validate="one_to_one")
    return df


_REGIONS = {}


def regions(cfg):
    """(fold, sample_idx, region): the Natural Earth continent of each test sample
    (identical across models: same folds, same samples)."""
    key = str(cfg["out_root_path"])
    if key not in _REGIONS:
        from shapfusion.analysis.basemap import assign_continents
        m0 = next(iter(cfg["models"]))
        parts = []
        for fold in cfg["folds"]:
            c = vcache.load(cfg, m0, fold)
            parts.append(pd.DataFrame({"fold": fold, "sample_idx": c["sample_idx"],
                                       "region": assign_continents(c["lon"], c["lat"])}))
        _REGIONS[key] = pd.concat(parts, ignore_index=True)
    return _REGIONS[key]


def phi_cols(views):
    return [f"phi_{short(v)}" for v in views]


def sii_cols(views):
    return [f"I_{pair_name(a, b)}" for a, b in pairs(views)]
