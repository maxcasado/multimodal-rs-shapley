"""
CropHarvest binary as a multi-view torch Dataset.

Adapted from code/datasets/{views_structure,utils}.py of fmenat/DSensDp. Behaviour kept:
  * every raw view is z-scored with the per-band statistics of data/stats/ (computed by the
    data provider over the whole dataset): x -> (x - mean) / std, float32;
  * a composite view ``A_B`` (S2_S2VI) is the per-date concatenation of the normalised raw
    views A and B along the last axis;
  * ``views_data_ident2indx`` maps an identifier to its row, per raw view; the Perceptual
    Score permutes a modality by swapping these maps (shapley/perceptual_score.py);
  * ``train_identifiers`` / ``val_identifiers`` and ``set_data_mode`` select the fold: the
    test identifiers keep the order of the split (the order of every stored prediction).
Changes: the views are held as numpy arrays (same values as the original xarray reads);
the train identifiers keep the file order (the original ``list(set(...))`` order depended on
PYTHONHASHSEED; the DataLoader shuffles them anyway); an optional ``perturbation`` (noise /
cloud gap, data/perturb.py) is applied to every item.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
import xarray as xr


def raw_views(views):
    """['S2_S2VI', 'S1'] -> ['S2', 'S2VI', 'S1']"""
    return [r for v in views for r in v.split("_")]


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, arrays: dict, identifiers, targets, stats: dict, views, coords=None):
        self.views_data = arrays                          # raw view -> (N, ...) float32
        self.identifiers = list(identifiers)
        self.views_data_ident2indx = {r: dict(zip(self.identifiers, range(len(self.identifiers))))
                                      for r in arrays}
        self.identifiers_target = dict(zip(self.identifiers, targets))
        self.stats = stats                                # raw view -> (mean, std)
        self.used_view_names = list(views)
        self.coords = coords                              # (N, 2) lon, lat in file order
        self.train_identifiers = list(self.identifiers)
        self.val_identifiers = []
        self.train_set = True
        self.perturbation = None

    # ── fold selection ───────────────────────────────────────────────────────
    def fold(self, test_ids, train: bool, perturbation=None) -> "MultiViewDataset":
        """A light copy restricted to the train or the test part of a fold (arrays shared)."""
        out = copy.copy(self)
        out.views_data_ident2indx = dict(self.views_data_ident2indx)
        test = set(test_ids)
        out.val_identifiers = list(test_ids)
        out.train_identifiers = [i for i in self.identifiers if i not in test]
        out.train_set = bool(train)
        out.perturbation = perturbation
        return out

    def get_all_identifiers(self) -> list:
        if len(self.val_identifiers) != 0:
            return self.train_identifiers if self.train_set else self.val_identifiers
        return self.train_identifiers

    def get_all_labels(self) -> np.ndarray:
        return np.array([self.identifiers_target[i] for i in self.get_all_identifiers()])

    def lonlat(self) -> np.ndarray:
        pos = self.views_data_ident2indx[raw_views(self.used_view_names)[0]]
        return self.coords[[pos[i] for i in self.get_all_identifiers()]]

    def get_view_shapes(self, view: str) -> tuple:
        shapes = [self.views_data[r].shape[1:] for r in view.split("_")]
        if len(shapes) == 1:
            return shapes[0]
        return shapes[0][:-1] + (sum(s[-1] for s in shapes),)

    def __len__(self) -> int:
        return len(self.get_all_identifiers())

    def _normalised(self, raw: str, ident) -> np.ndarray:
        x = self.views_data[raw][self.views_data_ident2indx[raw][ident]]
        mean, std = self.stats[raw]
        return ((x - mean) / std).astype("float32")

    def __getitem__(self, index: int) -> dict:
        ident = self.get_all_identifiers()[index]
        views = {}
        for v in self.used_view_names:
            parts = [self._normalised(r, ident) for r in v.split("_")]
            views[v] = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)
        item = {"identifier": ident, "views": views, "target": self.identifiers_target[ident]}
        if self.perturbation is not None:
            item = self.perturbation(item, index)
        return item


def load_dataset(data_dir, name: str, views, n_samples: int | None = None) -> MultiViewDataset:
    """data/<name>.nc + data/stats/stats_<name>.nc. ``n_samples`` keeps that many rows evenly
    spaced over the file (smoke runs: every sub-dataset, both classes)."""
    with xr.open_dataset(f"{data_dir}/{name}.nc", engine="h5netcdf") as ds:
        ds = ds.load()
        n = ds.sizes["identifier"]
        sel = (slice(None) if n_samples is None
               else np.unique(np.linspace(0, n - 1, int(n_samples)).round().astype(int)))
        ids = ds["identifier"].values[sel]
        arrays = {r: ds[r].values[sel] for r in raw_views(views)}
        targets = ds["target"].values[sel]
        coords = ds["coords"].values[sel]
        if not ds["train_mask"].values[sel].all():
            raise ValueError("expected train_mask all True (cross-validation on the whole set)")
    with xr.open_dataset(f"{data_dir}/stats/stats_{name}.nc", engine="h5netcdf") as st:
        st = st.load()
        stats = {r: (st[f"{r}-mean"].values, st[f"{r}-std"].values) for r in raw_views(views)}
    return MultiViewDataset(arrays, ids, targets, stats, views, coords)


def create_dataloader(dataset, batch_size: int = 32, train: bool = True, parallel_processes: int = 2):
    workers = max(int(len(os.sched_getaffinity(0)) / parallel_processes), 0)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                       shuffle=train, pin_memory=torch.cuda.is_available(),
                                       drop_last=False)
