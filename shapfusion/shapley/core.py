"""
Cooperative games over the sensors: exact Shapley values, order-2 Shapley interaction
index (Grabisch-Roubens), and the coalition inference that builds the games.

v-dict (one per job, vdict.pkl):
    {"format": "v2", "y_true": (N, 1), "task_type": "classification", "view_names": [...],
     "coalitions": {frozenset(S): record}}       16 coalitions, ∅ and N included
    record = {"pred": (N, C) softmax}             deterministic model (one forward pass)
           | {"metrics": {m: mean over R draws}, "metrics_std": {...}, "n_draws": R
              [, "p1_mean": (N,), "p0_mean": (N,)]}      stochastic model (EmbraceNet)
The ∅ record is the TRAIN-majority prediction; ``scalar_vdict`` replaces its f1_weighted by
the paper's v(∅) (characteristic.V_EMPTY_F1). For a stochastic model v(S) is the metric
averaged over the R draws; ``p1_mean`` / ``p0_mean`` (mean class probabilities over the same
draws) feed the per-sample games.

The game-theory functions work on {frozenset -> value} where the values may be scalars or
numpy arrays (one game per sample, elementwise).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .characteristic import CHAR_FUNCS, DEFAULT_METRIC, V_EMPTY_F1, char_value, train_baseline_record


# ── coalitions as bitmasks (bit k <-> views[k]) ─────────────────────────────
def mask_of(S, views) -> int:
    return sum(1 << views.index(v) for v in S)


def members(m: int, views) -> frozenset:
    return frozenset(v for k, v in enumerate(views) if m >> k & 1)


# ── v-dict access ────────────────────────────────────────────────────────────
def coalition_scalar(record: dict, metric: str, y_true) -> float:
    if "pred" in record:
        return char_value(y_true, record["pred"], metric)
    return float(record["metrics"][metric])


def scalar_vdict(v_dict: dict, metric: str = DEFAULT_METRIC, stored_empty: bool = False) -> dict:
    """{frozenset -> v(S)} under ``metric``; v(∅) = V_EMPTY_F1 for f1_weighted."""
    y = v_dict["y_true"]
    out = {S: coalition_scalar(rec, metric, y) for S, rec in v_dict["coalitions"].items()}
    if not stored_empty and metric == "f1_weighted" and frozenset() in out:
        out[frozenset()] = V_EMPTY_F1
    return out


# ── game theory ──────────────────────────────────────────────────────────────
def shapley_values(views: list, v: dict) -> dict:
    """Exact Shapley values {view: phi}."""
    n = len(views)
    out = {}
    for view in views:
        others = [x for x in views if x != view]
        phi = 0.0
        for size in range(n):
            w = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for S in itertools.combinations(others, size):
                S = frozenset(S)
                phi += w * (v[S | {view}] - v[S])
        out[view] = phi
    return out


def sii_weight(s: int, n: int) -> float:
    """Weight of Δ_ij(S), |S| = s: s!(n-s-2)!/(n-1)!  (n = 4: 1/3, 1/6, 1/3)."""
    return math.factorial(s) * math.factorial(n - s - 2) / math.factorial(n - 1)


def shapley_interaction_terms(views: list, v: dict) -> dict:
    """{(vi, vj): {s: Σ_{|S|=s, S ⊆ N∖{i,j}} w(s) Δ_ij(S)}}, i < j in ``views`` order, with
    Δ_ij(S) = v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)."""
    n = len(views)
    out = {}
    for a, vi in enumerate(views):
        for vj in views[a + 1:]:
            others = [x for x in views if x not in (vi, vj)]
            terms = {}
            for size in range(len(others) + 1):
                acc = 0.0
                for S in itertools.combinations(others, size):
                    S = frozenset(S)
                    acc = acc + (v[S | {vi, vj}] - v[S | {vi}] - v[S | {vj}] + v[S])
                terms[size] = sii_weight(size, n) * acc
            out[(vi, vj)] = terms
    return out


def shapley_interactions(views: list, v: dict) -> dict:
    """Order-2 Shapley interaction index {(vi, vj): I_ij}."""
    return {p: sum(t.values()) for p, t in shapley_interaction_terms(views, v).items()}


# ── building the game ────────────────────────────────────────────────────────
def run_subset_inference(predict, views: list, y_true, y_train, draws: int = 1, full_pred=None,
                         store_proba: bool = False, log=None) -> dict:
    """The v-dict of one fold.

    ``predict(subset, out_norm)`` returns the model's output on the test fold for the
    coalition ``subset`` ("softmax" probabilities, or raw logits for out_norm="").
    Deterministic model (draws = 1): one pass per coalition, softmax stored; ``full_pred``
    (the full-coalition softmax) is reused when given. Stochastic model: R passes per
    coalition, every metric of CHAR_FUNCS averaged over them; with ``store_proba`` also the
    mean class probabilities (computed in float64 from the logit difference).
    """
    stochastic = int(draws) > 1
    y_true = np.asarray(y_true)

    def record(subset):
        if not stochastic:
            return {"pred": np.asarray(predict(subset, "softmax"))}
        per = {m: [] for m in CHAR_FUNCS}
        s1 = s0 = 0.0
        for _ in range(int(draws)):
            z = np.asarray(predict(subset, ""), dtype=np.float64)
            d = z[:, 1] - z[:, 0]
            p1, p0 = 1.0 / (1.0 + np.exp(-d)), 1.0 / (1.0 + np.exp(d))
            proba = np.stack([p0, p1], axis=1)
            for m in CHAR_FUNCS:
                per[m].append(char_value(y_true, proba, m))
            s1, s0 = s1 + p1, s0 + p0
        rec = {"metrics": {m: float(np.mean(x)) for m, x in per.items()},
               "metrics_std": {m: float(np.std(x)) for m, x in per.items()},
               "n_draws": int(draws)}
        if store_proba:
            rec["p1_mean"], rec["p0_mean"] = s1 / draws, s0 / draws
        return rec

    coal = {frozenset(): train_baseline_record(y_train, y_true)}
    coal[frozenset(views)] = (record(list(views)) if stochastic or full_pred is None
                              else {"pred": np.asarray(full_pred)})
    for size in range(1, len(views)):
        for subset in itertools.combinations(views, size):
            coal[frozenset(subset)] = record(list(subset))
            if log:
                v = coalition_scalar(coal[frozenset(subset)], "f1_weighted", y_true)
                log(f"      {'+'.join(subset):<28} f1_weighted {v:.4f}")
    return {"format": "v2", "y_true": y_true, "task_type": "classification",
            "view_names": list(views), "coalitions": coal}
