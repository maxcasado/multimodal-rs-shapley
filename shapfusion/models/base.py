"""
Multi-sensor fusion models (PyTorch Lightning): one encoder per sensor, a fusion module, a
prediction head; inference on any coalition of sensors.

Adapted from code/models/{core_fusion,base_fusion,fusion_strategy,losses,missing_utils,
utils}.py of fmenat/DSensDp and of this project, pruned to what the paper's models use:
  * missing sensors are IGNORED (never imputed): only the encoders of the available sensors
    run and the fusion sees only their outputs;
  * DecisionFusion: each branch = encoder + head -> class logits, fusion -> Identity head;
    FeatureFusion: encoders -> fusion -> shared head;
  * sensor dropout during training (training.missing_as_aug), keep probability
    1 - random_perc per sensor:
      - deterministic fusions (mean): per-sample Bernoulli mask, masked mean of the branch
        outputs (= the fusion of the drawn coalition), loss on it;
      - stochastic fusions (EmbraceNet): one coalition per batch, forwarded through the real
        fusion (embracement) and head;
    the objective is main * CE(dropout prediction) + full * CE(all sensors), weights from
    ``weights_loss_variations`` (default main = 1, full = 0); validation = CE(all sensors).
"""
from typing import Dict, List

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

torch.set_float32_matmul_precision("high")          # as in the paper's runs (TF32 on Ampere)


def augment_random_missing(view_names, perc=0.0):
    """One coalition: each sensor kept with prob 1 - perc (numpy global RNG), redrawn until
    non-empty. (perc in {0, 1}: one of the 2^n - 1 non-empty coalitions, uniformly.)"""
    n = len(view_names)
    if perc != 0 and perc != 1:
        kept = []
        while len(kept) == 0:
            for v in view_names:
                if np.random.rand() > perc:
                    kept.append(v)
        return kept
    import itertools
    options = list(itertools.product([0, 1], repeat=n))[1:][::-1]
    mask = np.asarray(options[np.random.randint(0, 2 ** n - 1)]).astype(bool)
    return np.asarray(view_names)[mask].tolist()


def _detach(z):
    if isinstance(z, dict):
        return {k: _detach(v) for k, v in z.items()}
    return z.detach().cpu().numpy()


class MVFusionMissing(pl.LightningModule):
    def __init__(self, view_encoders: Dict[str, nn.Module], fusion_module: nn.Module,
                 prediction_head: nn.Module, loss_args: dict = {}, view_names: List[str] = [],
                 weights_loss_variations: dict = {}, optimizer: str = "adam", lr: float = 1e-3,
                 weight_decay: float = 0, **kwargs):
        super().__init__()
        self.optimizer, self.lr, self.weight_decay = optimizer, lr, weight_decay
        if isinstance(view_encoders, list):
            view_encoders = dict(zip(view_names, view_encoders))
        self.view_encoders = nn.ModuleDict(view_encoders)
        self.view_names = list(self.view_encoders.keys())
        self.fusion_module = fusion_module
        self.prediction_head = prediction_head
        self.N_views = len(self.view_encoders)
        self.weights_loss_variations = weights_loss_variations
        loss_args = {k: v for k, v in loss_args.items() if k not in ("name", "n_labels")}
        self.criteria = nn.CrossEntropyLoss(reduction="mean", **loss_args)
        self.missing_as_aug = False

    # ── optimisation ─────────────────────────────────────────────────────────
    def configure_optimizers(self):
        opt = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[self.optimizer]
        return opt(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        loss = self.loss_batch(batch)
        for k, v in loss.items():
            self.log("train_" + k, v, prog_bar=True)
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        loss = self.loss_batch(batch)
        for k, v in loss.items():
            self.log("val_" + k, v, prog_bar=True)
        return loss["objective"]

    def set_missing_info(self, aug_status, name: str = "ignore", random_perc=0, **kwargs):
        if name != "ignore":
            raise ValueError("only missing_method 'ignore' is used by the paper")
        self.missing_as_aug = aug_status
        self.missing_method = {"name": name, "where": "", "value_fill": None}
        self.random_perc = random_perc

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, views: Dict[str, torch.Tensor], out_norm: str = "", inference_views: list = [],
                missing_method: dict = {}, **kwargs) -> Dict[str, torch.Tensor]:
        """``inference_views`` = the coalition (all sensors when empty); the encoders of the
        other sensors do not run. ``out_norm``: "softmax" or "" (raw logits)."""
        present = self.view_names if len(inference_views) == 0 else inference_views
        if len(inference_views) != 0 and missing_method.get("name") != "ignore":
            raise ValueError("a coalition needs missing_method {'name': 'ignore'}")
        reps = {v: self.view_encoders[v](views[v]) for v in self.view_names if v in present and v in views}
        views_data = [reps[v] for v in self.view_names if v in present]
        available = (torch.ones(self.N_views) if len(inference_views) == 0
                     else torch.Tensor([1 if v in inference_views else 0 for v in self.view_names]))
        joint = self.fusion_module(views_data, views_available=available.bool())
        logits = self.prediction_head(joint["joint_rep"])
        pred = nn.Softmax(dim=-1)(logits) if out_norm == "softmax" else logits
        return {"prediction": pred, "last_layer": logits, "views:rep": reps, **joint}

    def prepare_batch(self, batch: dict, return_target=True):
        target = batch["target"].squeeze().to(torch.long) if return_target else None
        return batch["views"], target

    def _sample_training_coalition(self) -> list:
        """Batch-level sensor dropout for stochastic fusions: keep each sensor with prob
        1 - random_perc, at least one."""
        alpha = getattr(self, "random_perc", 0) or 0
        keep = [v for v in self.view_names if torch.rand(()).item() > alpha]
        if not keep:
            keep = [self.view_names[int(torch.randint(0, len(self.view_names), (1,)).item())]]
        return keep

    def loss_batch(self, batch: dict) -> dict:
        views, target = self.prepare_batch(batch)
        w = self.weights_loss_variations
        if self.missing_as_aug and self.training:
            out_full = self(views)
            full_pred = out_full["prediction"]
            if self.fusion_module.needs_stochastic_dropout():
                subset = self._sample_training_coalition()
                missing_pred = self(views, inference_views=subset,
                                    missing_method={"name": "ignore"})["prediction"]
            else:
                branch = torch.stack([out_full["views:rep"][v] for v in self.view_names], dim=1)
                B, S, _ = branch.shape
                mask = (torch.rand(B, S, device=branch.device) > self.random_perc).float()
                empty = mask.sum(dim=1) == 0
                if empty.any():
                    idx = torch.randint(0, S, (int(empty.sum()),), device=branch.device)
                    mask[empty, idx] = 1.0
                weights = mask / mask.sum(dim=1, keepdim=True)
                missing_pred = self.prediction_head((branch * weights.unsqueeze(-1)).sum(dim=1))
            loss_full = self.criteria(full_pred, target)
            return {"objective": w.get("main", 1) * self.criteria(missing_pred, target)
                    + w.get("full", 0) * loss_full,
                    "loss_full": loss_full, "full": loss_full,
                    "missing": self.criteria(missing_pred, target)}
        loss_full = self.criteria(self(views)["prediction"], target)
        return {"objective": w.get("main", 1) * loss_full, "loss_full": loss_full}

    # ── inference ────────────────────────────────────────────────────────────
    def transform(self, loader, out_norm: str = "", device: str = "", args_forward: dict = {},
                  **kwargs) -> dict:
        """Predictions over a DataLoader: {"prediction": (N, C)} (eval mode, no grad)."""
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.eval()
        self.to(device)
        outs = []
        with torch.no_grad():
            for batch in loader:
                views, _ = self.prepare_batch(batch, return_target=False)
                views = {k: v.to(device) for k, v in views.items()}
                outs.append(_detach(self(views, out_norm=out_norm, **args_forward)["prediction"]))
        self.train()
        return {"prediction": np.concatenate(outs, axis=0)}


class DecisionFusion(MVFusionMissing):
    """Branch = encoder + head (class logits); fusion of the logits; Identity head."""

    def __init__(self, view_encoders, fusion_module, loss_args: dict = {}, view_names=[], **kwargs):
        super().__init__(view_encoders, fusion_module, nn.Identity(), loss_args=loss_args,
                         view_names=view_names, **kwargs)
        self.save_hyperparameters(ignore=["view_encoders", "fusion_module"])


class FeatureFusion(MVFusionMissing):
    """Encoders -> fusion of the representations -> shared head."""

    def __init__(self, view_encoders, fusion_module, predictive_model, loss_args: dict = {},
                 view_names=[], **kwargs):
        super().__init__(view_encoders, fusion_module, predictive_model, loss_args=loss_args,
                         view_names=view_names, **kwargs)
        self.save_hyperparameters(ignore=["view_encoders", "fusion_module", "predictive_model"])
