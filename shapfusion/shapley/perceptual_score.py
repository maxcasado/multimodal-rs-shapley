"""
Permutation Perceptual Score (Gat et al., NeurIPS 2021):

    PS_raw(m) = f1(all sensors) - mean over n_perm permutations of f1(all | m permuted)

A modality is permuted across the test samples by permuting the identifier -> row map of
its raw views (every other modality and the label stay with the original sample).
Same implementation as the paper's runs.
"""
import numpy as np

from shapfusion.data.dataset import create_dataloader
from .characteristic import compute_metrics


def _permute(data_te, modality: str, perm_idx) -> dict:
    ids = data_te.val_identifiers
    backups = {}
    for raw in modality.split("_"):
        orig = data_te.views_data_ident2indx[raw]
        backups[raw] = orig
        new = dict(orig)
        for i, ident in enumerate(ids):
            new[ident] = orig[ids[perm_idx[i]]]
        data_te.views_data_ident2indx[raw] = new
    return backups


def perceptual_score(model, data_te, y_true, views, batch_size, f1_all, n_perm=10, seed=0,
                     metric="f1_weighted", device="") -> list:
    rng = np.random.default_rng(seed)
    n = len(data_te.val_identifiers)
    rows = []
    for modality in views:
        scores = []
        for _ in range(n_perm):
            backups = _permute(data_te, modality, rng.permutation(n))
            try:
                out = model.transform(create_dataloader(data_te, batch_size=batch_size, train=False),
                                      out_norm="softmax", device=device,
                                      args_forward={"inference_views": list(views),
                                                    "missing_method": model.missing_method})
            finally:
                data_te.views_data_ident2indx.update(backups)
            scores.append(compute_metrics(y_true, out["prediction"])[metric])
        rows.append({"modality": modality, "f1_all": f1_all,
                     "f1_permuted_mean": float(np.mean(scores)), "f1_permuted_std": float(np.std(scores)),
                     "PS_raw": f1_all - float(np.mean(scores)), "n_perm": n_perm})
    return rows
