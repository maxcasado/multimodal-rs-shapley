"""
DSensD+ (Mena et al., "Multi-sensor Model for Earth Observation Robust to Missing Data via
Sensor Dropout and Mutual Distillation", IEEE Access 13, 2025), as the reference code
fmenat/DSensDp @ 9d8c2ba with its example config.

Architecture = DecisionFusion: one branch per sensor (encoder -> MLP head -> C logits,
optionally LayerNorm(C) via ``use_norm_last``), identically initialised but unshared heads,
fusion = mean of the logits of the AVAILABLE sensors. Only the training objective differs.

Training step, two forward passes in train mode:
  1. every sensor                                   -> l_full = mean_v l_v, per-view l_v
  2. ONE coalition for the whole batch, drawn with numpy's global RNG after pass 1 (each
     sensor kept with prob 1 - random_perc, redrawn until non-empty); with
     ``faithful_single_view`` (the reference behaviour) only its FIRST sensor is forwarded
                                                    -> l_miss
  L = main CE_w(l_miss) + full CE_w(l_full)
      + individual    / V * sum_v CE_w(l_v)
      + individual_sd / V * sum_v T^2 KL( softmax(l_full / T).detach() || softmax(l_v / T) )
CE_w is the class-weighted CE, T = cross_temp. Validation: main * CE_w(l_full).

With C = 2 the LayerNorm variant maps each branch output to one of two fixed vectors, so
its encoders get no gradient: that model ("DSensD+ (LN)") is kept for reference only.
"""
import torch.nn.functional as F

from .base import DecisionFusion, augment_random_missing


class DSensDPlus(DecisionFusion):
    def __init__(self, view_encoders, fusion_module, loss_args: dict = {}, view_names: list = [],
                 faithful_single_view: bool = True, **kwargs):
        super().__init__(view_encoders, fusion_module, loss_args=loss_args, view_names=view_names,
                         **kwargs)
        self.faithful_single_view = faithful_single_view
        self.save_hyperparameters(ignore=["view_encoders", "fusion_module"])

    def sample_available(self) -> list:
        kept = augment_random_missing(self.view_names, perc=self.random_perc)
        return kept[:1] if self.faithful_single_view else kept

    def loss_batch(self, batch: dict) -> dict:
        views, target = self.prepare_batch(batch)
        w = self.weights_loss_variations
        out_full = self(views)
        full_pred, pred_views = out_full["prediction"], out_full["views:rep"]
        loss_full = self.criteria(full_pred, target)
        if self.missing_as_aug and self.training:
            available = self.sample_available()
            miss_pred = self(views, inference_views=available,
                             missing_method=self.missing_method)["prediction"]
            loss_miss = self.criteria(miss_pred, target)
            out = {"objective": w.get("main", 1) * loss_miss + w.get("full", 0) * loss_full,
                   "loss_full": loss_full, "full": loss_full, "missing": loss_miss}
        else:
            out = {"objective": w.get("main", 1) * loss_full, "loss_full": loss_full}
        if len(w) != 0 and self.training:
            T, V = w.get("cross_temp", 1), len(self.view_names)
            teacher = F.log_softmax(full_pred / T, dim=-1).detach()
            for v in self.view_names:
                if w.get("individual", 0) != 0:
                    out[v] = self.criteria(pred_views[v], target)
                    out["objective"] = out["objective"] + w["individual"] * out[v] / V
                if w.get("individual_sd", 0) != 0:
                    out[v + "-sd"] = (T ** 2) * F.kl_div(F.log_softmax(pred_views[v] / T, dim=-1),
                                                         teacher, log_target=True, reduction="batchmean")
                    out["objective"] = out["objective"] + w["individual_sd"] * out[v + "-sd"] / V
        return out
