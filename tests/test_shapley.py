"""Game theory on toy games with a known answer (n = 4 players named like the sensors),
Harsanyi identities, the paper's v(∅), and the v-dict builder."""
import itertools
from fractions import Fraction

import numpy as np
import pytest

from shapfusion.analysis import games as G
from shapfusion.shapley import core as sh
from shapfusion.shapley.characteristic import V_EMPTY_F1, uniform_v_empty_f1

V = ["S2_S2VI", "S1", "weather", "DEM"]
PAIRS = list(itertools.combinations(V, 2))
ALL = [frozenset(c) for s in range(5) for c in itertools.combinations(V, s)]
TOL = 1e-12


def game(fn):
    return {S: fn(S) for S in ALL}


def random_game(seed):
    rng = np.random.default_rng(seed)
    return {S: float(rng.normal()) for S in ALL}


def test_sii_weights_n4_are_third_sixth_third():
    assert [Fraction(sh.sii_weight(s, 4)).limit_denominator(100) for s in range(3)] == \
        [Fraction(1, 3), Fraction(1, 6), Fraction(1, 3)]


@pytest.mark.parametrize("ctx", [(), ("weather",), ("DEM",), ("weather", "DEM")])
def test_sii_weights_recovered_empirically(ctx):
    """v(T) = 1 iff T = ctx ∪ {S2, S1}: the only non-zero second difference of (S2, S1) is
    Δ(ctx) = 1, so I = w(|ctx|)."""
    target = frozenset(ctx) | {"S2_S2VI", "S1"}
    v = game(lambda S: 1.0 if S == target else 0.0)
    assert abs(sh.shapley_interactions(V, v)[("S2_S2VI", "S1")] - sh.sii_weight(len(ctx), 4)) < TOL


def test_additive_game():
    a = dict(zip(V, [0.7, -0.2, 0.35, 0.05]))
    v = game(lambda S: sum(a[i] for i in S))
    assert all(abs(sh.shapley_values(V, v)[i] - a[i]) < TOL for i in V)
    assert all(abs(x) < TOL for x in sh.shapley_interactions(V, v).values())


def test_pure_redundancy():
    """v(S) = 1 if S ≠ ∅: φ_i = 1/4, I_ij = (1/3)(1 - 1 - 1 + 0) = -1/3."""
    v = game(lambda S: 1.0 if S else 0.0)
    assert all(abs(p - 0.25) < TOL for p in sh.shapley_values(V, v).values())
    assert all(abs(x + 1 / 3) < TOL for x in sh.shapley_interactions(V, v).values())


def test_pure_complementarity():
    """v(S) = 1 iff {S2, S1} ⊆ S: I_{S2,S1} = 1, 0 elsewhere; φ = 1/2, 1/2, 0, 0."""
    v = game(lambda S: 1.0 if {"S2_S2VI", "S1"} <= S else 0.0)
    I = sh.shapley_interactions(V, v)
    assert abs(I[("S2_S2VI", "S1")] - 1.0) < TOL
    assert all(abs(x) < TOL for p, x in I.items() if p != ("S2_S2VI", "S1"))
    phi = sh.shapley_values(V, v)
    assert [round(phi[i], 12) for i in V] == [0.5, 0.5, 0.0, 0.0]


def test_array_games_match_scalar_games():
    rng = np.random.default_rng(0)
    arr = {S: rng.normal(size=20) for S in ALL}
    phi, I = sh.shapley_values(V, arr), sh.shapley_interactions(V, arr)
    for k in range(20):
        vk = {S: x[k] for S, x in arr.items()}
        assert all(abs(phi[i][k] - sh.shapley_values(V, vk)[i]) < TOL for i in V)
        assert all(abs(I[p][k] - sh.shapley_interactions(V, vk)[p]) < TOL for p in PAIRS)
    assert np.abs(sum(phi.values()) - (arr[frozenset(V)] - arr[frozenset()])).max() < TOL   # efficiency


@pytest.mark.parametrize("seed", range(5))
def test_harsanyi_identities(seed):
    v = random_game(seed)
    err = G.identity_errors(v)
    assert err["I"] < TOL and err["phi"] < TOL
    d = G.harsanyi(v)
    for S in ALL:                                       # Möbius inversion
        assert v[S] - v[frozenset()] == pytest.approx(sum(x for T, x in d.items() if T <= S), abs=TOL)
    for (i, j), (I, Iv, Ic) in G.interaction_split(v).items():
        e = frozenset()
        assert I == pytest.approx(Iv + Ic, abs=TOL)
        assert Iv == pytest.approx((v[e | {i, j}] - v[e | {i}] - v[e | {j}] + v[e]) / 3, abs=TOL)


def test_decomposition_sums_to_phi():
    v = random_game(3)
    for m, d in G.decomposition(v).items():
        assert d["phi"] == pytest.approx(d["phi_grand"] + d["phi_small"], abs=TOL)


def test_uniform_v_empty():
    assert V_EMPTY_F1 == pytest.approx(0.5127, abs=1e-4)
    p = 0.37                                           # coin-flip predictions, expected weighted F1
    f1 = [2 * (s / 2) / (2 * (s / 2) + (1 - s) / 2 + s / 2) for s in (p, 1 - p)]
    assert uniform_v_empty_f1(p) == pytest.approx(p * f1[0] + (1 - p) * f1[1], abs=TOL)


def test_scalar_vdict_uses_paper_empty():
    vd = {"y_true": np.array([0, 1, 1]),
          "coalitions": {frozenset(): {"pred": np.array([[0, 1]] * 3, float)},
                         frozenset(["S1"]): {"pred": np.array([[1, 0], [0, 1], [0, 1]], float)}}}
    assert sh.scalar_vdict(vd)[frozenset()] == V_EMPTY_F1
    assert sh.scalar_vdict(vd, stored_empty=True)[frozenset()] == pytest.approx(0.8 * 2 / 3, abs=1e-12)
    assert sh.scalar_vdict(vd)[frozenset(["S1"])] == pytest.approx(1.0)


def test_run_subset_inference_deterministic_and_stochastic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(50, 1))
    logits = {frozenset(S): rng.normal(size=(50, 2)) for S in ALL if S}

    def predict(subset, out_norm="softmax"):
        z = logits[frozenset(subset)] + (rng.normal(scale=0.1, size=(50, 2)) if stochastic else 0)
        return np.exp(z) / np.exp(z).sum(1, keepdims=True) if out_norm == "softmax" else z

    stochastic = False
    vd = sh.run_subset_inference(predict, V, y, y_train=np.array([1, 1, 0]))
    assert len(vd["coalitions"]) == 16 and "pred" in vd["coalitions"][frozenset(V)]
    assert (vd["coalitions"][frozenset()]["pred"].argmax(1) == 1).all()      # TRAIN majority
    stochastic = True
    vd = sh.run_subset_inference(predict, V, y, y_train=np.array([0]), draws=4, store_proba=True)
    rec = vd["coalitions"][frozenset(["S1", "DEM"])]
    assert rec["n_draws"] == 4 and np.allclose(rec["p1_mean"] + rec["p0_mean"], 1.0)
    assert 0 <= sh.scalar_vdict(vd)[frozenset(["S1"])] <= 1
