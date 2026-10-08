"""Build a model from its recipe (configs/models/<model>.yaml) and the training data.

Adapted from code/training/learn_pipeline.py (MultiFusion_train / build_model) and
code/training/utils.py (assign_labels_weights) of fmenat/DSensDp."""
import copy

import numpy as np
import torch
from sklearn.utils import class_weight

from .base import DecisionFusion, FeatureFusion
from .dsensdp import DSensDPlus
from .encoders import create_model
from .fusion import FusionModuleMissing


def class_weights(labels) -> dict:
    """{"n_labels", "weight"}: class-balanced CE weights of the training labels."""
    y = np.asarray(labels).astype(int).flatten()
    return {"n_labels": int(y.max() + 1),
            "weight": class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)}


def input_dim(dataset, view: str, encoder_spec: dict):
    shape = dataset.get_view_shapes(view)
    if encoder_spec["model_type"] == "mlp":
        return int(np.prod(shape))
    return shape[-1] if len(shape) < 3 else int(np.prod(shape[1:]))


def build_model(dataset, model_cfg: dict, loss_weights: dict):
    """The (untrained) model; ``loss_weights`` from class_weights(train labels)."""
    training, arch, method = model_cfg["training"], model_cfg["architecture"], copy.deepcopy(model_cfg["method"])
    n_labels = loss_weights["n_labels"]
    loss_args = {"name": training.get("loss_args", {}).get("name", "ce"),
                 "weight": torch.tensor(loss_weights["weight"], dtype=torch.float)}
    emb_dim = training["emb_dim"]
    views = list(dataset.used_view_names)
    encoders = {v: create_model(input_dim(dataset, v, arch["encoders"][v]), emb_dim, **arch["encoders"][v])
                for v in views}
    args = {"loss_args": loss_args, **training.get("additional_args", {})}
    if method["feature"]:
        fusion = FusionModuleMissing([e.get_output_size() for e in encoders.values()], **method["agg_args"])
        head = create_model(fusion.get_info_dims()["joint_dim"], n_labels, **arch["predictive_model"], encoder=False)
        model = FeatureFusion(encoders, fusion, head, view_names=views, **args)
    else:
        fusion = FusionModuleMissing([n_labels] * len(encoders), **method["agg_args"])
        head = create_model(emb_dim, n_labels, **arch["predictive_model"], encoder=False)
        branches = {}
        for v in views:                        # identically initialised, unshared heads
            h = copy.deepcopy(head)
            h.load_state_dict(head.state_dict())
            branches[v] = torch.nn.Sequential(encoders[v], h)
        spec = method.get("dsensd_plus")
        if spec:
            model = DSensDPlus(branches, fusion, view_names=views, **args, **dict(spec))
        else:
            model = DecisionFusion(branches, fusion, view_names=views, **args)
    model.set_missing_info(training.get("missing_as_aug", False), **training["missing_method"])
    return model
