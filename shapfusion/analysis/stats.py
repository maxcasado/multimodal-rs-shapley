"""Cross-fold aggregation: mean, std (ddof = 1), SNR = mean / std; a value is reported as
unresolved (n.r.) when |SNR| < 2."""
from __future__ import annotations

import numpy as np
import pandas as pd

SNR_THRESHOLD = 2.0


def agg(values, ddof: int = 1) -> dict:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "snr": np.nan, "n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=ddof)) if n > 1 else 0.0
    snr = (np.inf * np.sign(mean) if mean != 0 else np.nan) if std == 0.0 else mean / std
    return {"mean": mean, "std": std, "snr": snr, "n": int(n)}


def summary(df: pd.DataFrame, keys: list, cols, inf_resolved: bool = False) -> pd.DataFrame:
    """Per group of ``keys``: <col>_mean, _std, _snr, _unresolved over the folds (a zero std,
    SNR = ±inf, counts as resolved only with ``inf_resolved``)."""
    rows = []
    for k, sub in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        for c in cols:
            a = agg(sub[c].values)
            row.update({f"{c}_mean": a["mean"], f"{c}_std": a["std"], f"{c}_snr": a["snr"],
                        f"{c}_unresolved": not ((inf_resolved or np.isfinite(a["snr"]))
                                                and abs(a["snr"]) >= SNR_THRESHOLD)})
        rows.append(row)
    return pd.DataFrame(rows)
