"""The 5-fold split of the paper: np.random.seed(10), shuffle the identifiers (file order),
np.array_split into 5. Every experiment, model and level uses the same folds."""
from __future__ import annotations

import hashlib

import numpy as np

from shapfusion import config as C


def make_splits(identifiers, seed: int, k: int) -> list:
    ids = np.array(identifiers, copy=True)
    np.random.seed(seed)
    np.random.shuffle(ids)
    return np.array_split(ids, k)


def get_splits(cfg: dict, dataset) -> list:
    """Test identifiers of each fold, written once to <runs_dir>/splits.npz."""
    path = C.splits_file(cfg)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            folds = [z[f"fold{f}"] for f in range(cfg["cv"]["k"])]
        if sum(len(f) for f in folds) != len(dataset.identifiers):
            raise ValueError(f"{path} does not match the dataset ({len(dataset.identifiers)} samples)")
        return folds
    folds = make_splits(dataset.identifiers, cfg["cv"]["seed"], cfg["cv"]["k"])
    digest = hashlib.sha256("|".join("|".join(map(str, f)) for f in folds).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sha256=digest, **{f"fold{f}": folds[f].astype(str) for f in range(len(folds))})
    return folds
