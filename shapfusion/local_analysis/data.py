"""Shared in-process cache of the per-sample frames and per-cell statistics, so the
analysis steps of one report run do not recompute them. Also the common writers
(table + key numbers)."""
import json
from pathlib import Path

import numpy as np

from . import games, grid

_FRAMES = {}
_STATS = {}


def frame(cfg, model, vf="prob_pred"):
    key = (str(cfg["out_root_path"]), model, vf, cfg.get("tag", ""))
    if key not in _FRAMES:
        df = games.sample_frame(cfg, model, vf, with_terms=True)
        _FRAMES[key] = grid.add_cells(df, cfg)
    return _FRAMES[key]


def stats(cfg, model, col, vf="prob_pred"):
    key = (str(cfg["out_root_path"]), model, col, vf, cfg.get("tag", ""), cfg["grid"]["res"], cfg["grid"]["n_min"],
           cfg["grid"]["min_per_fold"], cfg["grid"]["sign_k"])
    if key not in _STATS:
        _STATS[key] = grid.cell_stats(frame(cfg, model, vf), col, cfg)
    return _STATS[key]


def quantities(cfg):
    views = cfg["views"]
    return games.phi_cols(views), games.sii_cols(views)


def qlabel(col):
    """Display name of a column: phi_S2 -> S2, I_S2xweather -> S2 × weather."""
    if col.startswith("phi_"):
        return col[4:]
    if col.startswith("I_"):
        return col[2:].replace("x", " × ", 1) if col.count("x") == 1 else \
            " × ".join(col[2:].split("x"))
    return col


def model_label(cfg, model, quantised_note=None):
    spec = cfg["models"][model]
    lab = spec["label"]
    if spec.get("quantised") and quantised_note:
        lab += f"\n{quantised_note}"
    return lab


def write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=conv))
    return path


def mean_std(values):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else np.nan


def fmt(m, s, d=3):
    return f"{m:+.{d}f} ± {s:.{d}f}" if np.isfinite(s) else f"{m:+.{d}f}"
