"""The GeoJSON of the local Shapley values on fake caches: valid FeatureCollection, dataset
order, coordinates, and efficiency of the rounded phi (sum = P(ŷ_full | x) − 0.5)."""
import json

import numpy as np

from shapfusion import config as C
from shapfusion.local_analysis import games as LG
from shapfusion.report import geojson

SENSORS = ["S2", "S1", "weather", "DEM"]


def fake_cache(n, fold):
    rng = np.random.default_rng(fold)
    idx = np.arange(fold, 2 * n, 2)                       # the two folds interleave in the dataset
    return {"p1": rng.uniform(0.01, 0.99, size=(16, n)), "y_true": rng.integers(0, 2, n), "sample_idx": idx,
            "identifier": np.array([f"{i}_fake-dataset" for i in idx]),
            "lat": rng.uniform(-50, 60, n), "lon": rng.uniform(-170, 170, n)}


def test_geojson(monkeypatch, tmp_path):
    cfg = C.load()
    cfg["folds"] = [0, 1]
    caches = {f: fake_cache(50, f) for f in cfg["folds"]}
    monkeypatch.setattr(LG.vcache, "load", lambda la, model, fold: caches[fold])
    back = json.loads(geojson.write(geojson.feature_collection(cfg), tmp_path / "x.geojson").read_text())
    assert back["type"] == "FeatureCollection" and len(back["features"]) == 100
    nd = cfg["geojson"]["decimals"]
    for i, f in enumerate(back["features"]):
        c, k = caches[i % 2], i // 2
        p = f["properties"]
        assert p["identifier"] == f"{i}_fake-dataset" and p["fold"] == i % 2
        assert f["geometry"]["coordinates"] == [round(c["lon"][k], 6), round(c["lat"][k], 6)]
        for m in cfg["geojson"]["models"]:
            p_full = c["p1"][15, k]
            assert p[f"{m}_yhat"] == int(p_full > 0.5)
            phi = sum(p[f"{m}_phi_{s}"] for s in SENSORS)
            assert abs(phi - (max(p_full, 1 - p_full) - 0.5)) <= 4 * 0.5 * 10 ** -nd + 1e-12
