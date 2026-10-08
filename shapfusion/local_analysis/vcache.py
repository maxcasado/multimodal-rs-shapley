"""Per-sample games of the main experiment, written by the ``local`` stage of each job
(runs/<model>/main/fold<k>/local.npz, see shapfusion/job.py)."""
import numpy as np

from shapfusion import config as C
from shapfusion.shapley.core import mask_of, members

__all__ = ["cache_file", "vdict_file", "load", "mask_of", "members"]


def cache_file(la, model, fold):
    return C.job_dir(la["paper"], model, "main", None, fold) / "local.npz"


def vdict_file(la, model, fold):
    return C.job_dir(la["paper"], model, "main", None, fold) / "vdict.pkl"


def load(la, model, fold) -> dict:
    with np.load(cache_file(la, model, fold), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
