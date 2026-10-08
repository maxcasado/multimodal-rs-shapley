"""Per-sample games: closed form of the interaction index for a mean-of-logits fusion, and the
local interaction step on a fake cache (Harsanyi identity, efficiency, orientation)."""
import numpy as np
import pytest

from shapfusion.analysis import games as G
from shapfusion.local_analysis import games, p0
from shapfusion.report import interactions_local as L

VIEWS = ["S2_S2VI", "S1", "weather", "DEM"]


def test_closed_form_algebra():
    """For ANY branch values and ANY b, the closed form equals core.py on the mean-of-branches game."""
    rng = np.random.default_rng(0)
    l = {v: rng.normal(scale=5, size=500) for v in VIEWS}
    for b in (0.0, 0.7, -2.3):
        T = np.empty((16, 500))
        T[0] = b
        for m in range(1, 16):
            T[m] = np.mean([l[v] for k, v in enumerate(VIEWS) if m >> k & 1], axis=0)
        I_sh = games.sii(T, VIEWS)
        cf = p0.closed_form_sii(l, VIEWS, b)
        assert np.abs(I_sh - np.stack([cf[p] for p in games.pairs(VIEWS)], axis=1)).max() < 1e-12


def fake_cache(n, seed):
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(0.01, 0.99, size=(16, n))
    p1[0] = np.nan
    return {"p1": p1, "y_true": rng.integers(0, 2, n), "sample_idx": np.arange(n),
            "lat": rng.uniform(-50, 60, n), "lon": rng.uniform(-170, 170, n)}


@pytest.fixture
def frame(monkeypatch):
    caches = {f: fake_cache(200, f) for f in range(2)}
    monkeypatch.setattr(L.vcache, "load", lambda cfg, model, fold: caches[fold])
    return L.sample_frame({"folds": [0, 1]}, "CoM")


def test_harsanyi_and_efficiency(frame):
    df, tables = frame
    checks = L.check(df, tables, np.random.default_rng(0))
    assert checks["I_via_harsanyi"] < 1e-12 and checks["efficiency"] < 1e-12


def test_split_and_orientation(frame):
    df, tables = frame
    T = np.concatenate(tables, axis=1)
    k = 7
    g = {L.vcache.members(m, L.V): float(T[m, k]) for m in range(16)}
    assert g[frozenset()] == 0.5
    assert g[frozenset(L.V)] >= 0.5                      # oriented on the full prediction
    for i, j in G.PAIRS:
        vide = (g[frozenset({i, j})] - g[frozenset({i})] - g[frozenset({j})] + 0.5) / 3
        assert df[L.col((i, j))].iat[k] - df[L.col((i, j), "I_ctx")].iat[k] == pytest.approx(vide, abs=1e-12)
