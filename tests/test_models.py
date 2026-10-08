"""The four models build with the exact state_dict layout (keys and shapes) of the
checkpoints that produced the paper, and run on any coalition of sensors."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from shapfusion import config as C
from shapfusion.data.dataset import MultiViewDataset
from shapfusion.models.build import build_model, class_weights

SIG = json.loads((Path(__file__).parent / "data" / "state_dict_signature.json").read_text())
VIEWS = ["S2_S2VI", "S1", "weather", "DEM"]
SHAPES = {"S2": (12, 11), "S2VI": (12, 1), "S1": (12, 2), "weather": (12, 2), "DEM": (2,)}


def fake_dataset(n=64, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.array([f"{i}_toy" for i in range(n)])
    arrays = {k: rng.normal(size=(n,) + s).astype("float32") for k, s in SHAPES.items()}
    stats = {k: (np.zeros(s[-1], "float32"), np.ones(s[-1], "float32")) for k, s in SHAPES.items()}
    targets = rng.integers(0, 2, size=(n, 1)).astype("int8")
    return MultiViewDataset(arrays, ids, targets, stats, VIEWS, coords=rng.uniform(-50, 50, (n, 2)))


@pytest.mark.parametrize("model", ["CoM", "EmbraceNet", "DSensDp_noLN", "DSensDp"])
def test_state_dict_matches_paper_checkpoints(model):
    cfg = C.load()
    ds = fake_dataset()
    m = build_model(ds, C.model_config(cfg, model), class_weights(ds.get_all_labels()))
    got = {k: list(v.shape) for k, v in m.state_dict().items()}
    assert got == SIG[model]


@pytest.mark.parametrize("model", ["CoM", "EmbraceNet"])
def test_every_coalition_runs(model):
    cfg = C.load()
    ds = fake_dataset()
    m = build_model(ds, C.model_config(cfg, model), class_weights(ds.get_all_labels())).eval()
    batch = {v: torch.from_numpy(np.stack([ds[i]["views"][v] for i in range(8)])) for v in VIEWS}
    for k in range(1, 5):
        for S in itertools.combinations(VIEWS, k):
            p = m(batch, out_norm="softmax", inference_views=list(S), missing_method={"name": "ignore"})["prediction"]
            assert p.shape == (8, 2) and torch.allclose(p.sum(1), torch.ones(8))


def test_decision_fusion_is_mean_of_branch_logits():
    cfg = C.load()
    ds = fake_dataset()
    m = build_model(ds, C.model_config(cfg, "CoM"), class_weights(ds.get_all_labels())).eval()
    batch = {v: torch.from_numpy(np.stack([ds[i]["views"][v] for i in range(8)])) for v in VIEWS}
    with torch.no_grad():
        branch = {v: m.view_encoders[v](batch[v]) for v in VIEWS}
        S = ["S2_S2VI", "weather", "DEM"]
        fused = m(batch, inference_views=S, missing_method={"name": "ignore"})["prediction"]
    assert torch.allclose(fused, torch.stack([branch[v] for v in S]).mean(0), atol=1e-6)
