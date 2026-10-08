"""
Fusion of the sensor representations, for the fusions of the paper.

Adapted from code/models/fusion_module.py and single/fusion_layers.py of fmenat/DSensDp
(pruned to the modes used here; attribute names unchanged):
  avg / avg_ignore   mean over the AVAILABLE sensors (CoM, DSensD+ at decision level);
                     ``_ignore`` uses nanmean
  embracenet         EmbraceNet (Choi & Lee, Inf. Fusion 2019): one docking Linear + ReLU
                     per sensor, then the embracement: every output component is copied
                     from one available sensor drawn uniformly (stochastic, also at test time)
Absent sensors are never imputed: the caller passes only the representations of the
available sensors and ``views_available`` (one flag per sensor, in view order).
"""
from typing import Dict, List

import torch
from torch import nn


class UniformSum_(nn.Module):
    def __init__(self, ignore: bool = False):
        super().__init__()
        self.ignore = ignore

    def forward(self, x, **kwargs):
        return {"rep": torch.nanmean(x, dim=1) if self.ignore else torch.mean(x, dim=1)}


class Stacking_(nn.Module):
    def forward(self, x, **kwargs):
        return torch.stack(x, dim=1)


class FusionModuleMissing(nn.Module):
    def __init__(self, emb_dims, mode: str, adaptive: bool = False, **kwargs):
        super().__init__()
        if adaptive:
            raise ValueError("adaptive fusion is not used by the paper")
        self.mode = mode
        self.base = mode.split("_")[0]
        self.emb_dims = list(emb_dims.values()) if isinstance(emb_dims, dict) else list(emb_dims)
        self.N_views = len(self.emb_dims)
        if len(set(self.emb_dims)) != 1:
            raise ValueError("all sensors must share the representation size")
        self.joint_dim = self.emb_dims[0]
        self.stacker_function = Stacking_()
        if self.base in ("avg", "mean"):
            self.pooler_function = UniformSum_(ignore=mode.split("_")[-1] == "ignore")
        elif self.base in ("embracenet", "embrace"):
            self.features = True
            c = kwargs.get("embrace_dim", self.joint_dim)
            self.joint_dim = c
            self.docking = nn.ModuleList([nn.Linear(int(d), c) for d in self.emb_dims])
            self.docking_activation = nn.ReLU()
        else:
            raise ValueError(f"fusion mode {mode!r} is not used by the paper")

    def forward(self, views_emb: List[torch.Tensor], views_available: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.base in ("embracenet", "embrace"):
            present = (torch.where(views_available.bool())[0].tolist() if len(views_available) > 0
                       else list(range(self.N_views)))
            views_emb = [self.docking_activation(self.docking[present[j]](views_emb[j]))
                         for j in range(len(views_emb))]
        stacked = self.stacker_function(views_emb)                     # (B, n_present, D)
        if self.base in ("avg", "mean"):
            return {"joint_rep": self.pooler_function(stacked)["rep"]}
        # embracement: p uniform over the present sensors (only those are passed in)
        n_batch, n_views, n_dims = stacked.shape
        p = torch.ones(n_views, dtype=torch.float, device=stacked.device)
        p = (p / p.sum()).repeat(n_batch, 1)
        selection = torch.multinomial(p, n_dims, replacement=True)[:, None, :]
        joint = stacked.gather(1, selection.expand(n_batch, n_views, n_dims))[:, 0, :]
        return {"joint_rep": joint}

    def get_info_dims(self):
        return {"emb_dims": self.emb_dims, "joint_dim": self.joint_dim, "feature_pool": True}

    def needs_stochastic_dropout(self) -> bool:
        """Stochastic fusions train sensor dropout through the real fusion (see base.py)."""
        return self.base in ("embracenet", "embrace")
