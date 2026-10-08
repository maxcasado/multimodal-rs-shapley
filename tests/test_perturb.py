"""The two S2 degradations: Gaussian noise keyed on (sigma, fold, position), cloud gaps nested
across levels, linear interpolation exact on observed dates."""
import numpy as np

from shapfusion.data.perturb import CloudGap, GaussianNoise, interpolate_missing, keep_indices


def item(seed=0):
    rng = np.random.default_rng(seed)
    return {"identifier": "17_kenya", "views": {"S2_S2VI": rng.normal(size=(12, 12)).astype("float32"),
                                                 "S1": rng.normal(size=(12, 2)).astype("float32")}}


def test_noise_is_deterministic_and_only_on_target():
    a, b = item(), item()
    na = GaussianNoise("S2_S2VI", 0.5, fold=2, seed_offset=999983)(a, index=7)
    nb = GaussianNoise("S2_S2VI", 0.5, fold=2, seed_offset=999983)(b, index=7)
    assert np.array_equal(na["views"]["S2_S2VI"], nb["views"]["S2_S2VI"])
    assert np.array_equal(na["views"]["S1"], item()["views"]["S1"])
    assert not np.array_equal(na["views"]["S2_S2VI"], item()["views"]["S2_S2VI"])
    other = GaussianNoise("S2_S2VI", 0.5, fold=2, seed_offset=999983)(item(), index=8)
    assert not np.array_equal(other["views"]["S2_S2VI"], na["views"]["S2_S2VI"])


def test_zero_levels_are_no_ops():
    for pert in (GaussianNoise("S2_S2VI", 0.0, 0, 1), CloudGap("S2_S2VI", 0, 12, 1)):
        out = pert(item(), index=3)
        assert np.array_equal(out["views"]["S2_S2VI"], item()["views"]["S2_S2VI"])


def test_cloud_gap_masks_are_nested():
    kept = [set(keep_indices("17_kenya", k, 12, 20250831)) for k in range(13)]
    assert all(len(kept[k]) == 12 - k for k in range(13))
    assert all(kept[k + 1] <= kept[k] for k in range(12))


def test_interpolation():
    arr = np.arange(12, dtype=float)[:, None] * np.array([[1.0, -2.0]])
    keep = np.array([0, 3, 4, 11])
    out = interpolate_missing(arr, keep)
    assert np.allclose(out, arr)                         # a linear series is rebuilt exactly
    edge = interpolate_missing(arr, np.array([2, 5]))
    assert np.allclose(edge[:2], arr[2]) and np.allclose(edge[6:], arr[5])     # edge-clamped
    assert np.array_equal(interpolate_missing(arr, np.array([], int)), np.zeros_like(arr))
    out = CloudGap("S2_S2VI", 12, 12, 20250831)(item(), index=0)
    assert np.array_equal(out["views"]["S2_S2VI"], np.zeros((12, 12), "float32"))
