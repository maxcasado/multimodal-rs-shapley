"""
The local Shapley values as a GeoJSON file, versioned at the root of the repository
(shapley_local.geojson, ``geojson`` block of configs/paper.yaml; under paper_outputs_smoke/
with --smoke).

One Point per test sample of the main experiment (lon/lat, WGS84), in dataset order; each
sample is explained once, by the model of its fold. Properties:

    identifier              CropHarvest identifier
    fold                    test fold of the sample
    y_true                  label, 1 = crop
    <model>_yhat            class predicted with the four sensors
    <model>_phi_<sensor>    local Shapley value, <sensor> in S2 (S2 + NDVI), S1, weather, DEM

for every model of ``geojson.models``. The game is that of the local maps (local_analysis/
games.py, value function prob_pred): v_x(S) = P(ŷ_full(x) | x_S), v_x(∅) = 0.5, so φ > 0 means
the sensor supports the class the model predicts, and the four φ of a model sum to
P(ŷ_full(x) | x) − 0.5.
"""
import json

import numpy as np

from shapfusion.local_analysis import config as la_config, games as LG

COORD_DECIMALS = 6        # ~0.1 m
DESCRIPTION = ("Local Shapley values of the four sensors (S2 + NDVI, S1, weather, DEM) of crop / non-crop "
               "classifiers on CropHarvest, one point per sample, each sample explained by the model of its "
               "test fold. Game: v_x(S) = P(yhat_full(x) | x_S), v_x(empty) = 0.5, so phi > 0 supports the "
               "class predicted with all four sensors. Properties: identifier, fold, y_true and, for each "
               "model in {models}, <model>_yhat and <model>_phi_<sensor>.")


def feature_collection(cfg) -> dict:
    spec = cfg["geojson"]
    la = la_config.build(cfg)
    cols = LG.phi_cols(la["views"])
    frames = {m: LG.sample_frame(la, m, "prob_pred", with_regions=False).sort_values("sample_idx", ignore_index=True)
              for m in spec["models"]}
    base = frames[spec["models"][0]]
    props = {"identifier": base["identifier"].tolist(), "fold": base["fold"].tolist(),
             "y_true": base["y_true"].tolist()}
    for m, df in frames.items():
        if not np.array_equal(df["sample_idx"].values, base["sample_idx"].values):
            raise ValueError(f"{m} does not explain the same test samples as {spec['models'][0]}")
        props[f"{m}_yhat"] = df["yhat"].tolist()
        for c in cols:
            props[f"{m}_{c}"] = df[c].round(int(spec["decimals"])).tolist()
    lon, lat = (base[k].round(COORD_DECIMALS).tolist() for k in ("lon", "lat"))
    return {"type": "FeatureCollection", "name": "shapley_local",
            "description": DESCRIPTION.format(models=", ".join(spec["models"])),
            "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon[i], lat[i]]},
                          "properties": {k: v[i] for k, v in props.items()}} for i in range(len(base))]}


def write(fc, path):
    """One feature per line: readable, and a rerun gives a line-wise git diff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = json.dumps({k: v for k, v in fc.items() if k != "features"}, ensure_ascii=False)
    feats = ",\n".join(json.dumps(f, separators=(",", ":"), allow_nan=False) for f in fc["features"])
    path.write_text(f'{head[:-1]}, "features": [\n{feats}\n]}}\n', encoding="utf-8")
    return path


def run(cfg):
    fc = feature_collection(cfg)
    path = write(fc, cfg["geojson_path"])
    print(f"    {path}: {len(fc['features'])} samples, {path.stat().st_size / 1e6:.1f} MB")
