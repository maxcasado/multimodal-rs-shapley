"""
Regular lon/lat grid (same binning as shapfusion/analysis/local_maps.grid_mean) and the
per-cell statistics every map of the local analyses is drawn from.

For a per-sample quantity x and a cell c:
    n            samples in c (all folds);  n_f<k>  samples of fold k in c
    mean         pooled mean over the n samples            (what the local maps show)
    abs_mean     pooled mean of |x|
    f<k>         mean over the samples of fold k (NaN if n_f<k> < min_per_fold)
    fold_mean    mean of the available fold means;  fold_std  their std (ddof=1, >= 2 folds)
    ratio        fold_mean / fold_std
    n_pos/n_neg  folds whose mean is > 0 / < 0;  sign_consistent  max(n_pos, n_neg) >= sign_k

The five folds PARTITION the test samples: a fold mean is over different samples AND a
different trained model, so the across-fold spread mixes model variability with the
sampling noise of a handful of samples per cell.
"""
import numpy as np
import pandas as pd


def edges(res):
    lon_e = np.arange(-180, 180 + res, res)
    lat_e = np.arange(-90, 90 + res, res)
    return lon_e, lat_e


def shape(res):
    lon_e, lat_e = edges(res)
    return len(lon_e) - 1, len(lat_e) - 1


def centres(res):
    lon_e, lat_e = edges(res)
    return (lon_e[:-1] + lon_e[1:]) / 2, (lat_e[:-1] + lat_e[1:]) / 2


def cell_index(lon, lat, res):
    """cell_id = i_lon * n_lat + i_lat (np.digitize binning of local_maps.grid_mean)."""
    lon_e, lat_e = edges(res)
    n_lon, n_lat = len(lon_e) - 1, len(lat_e) - 1
    il = np.clip(np.digitize(lon, lon_e) - 1, 0, n_lon - 1)
    jl = np.clip(np.digitize(lat, lat_e) - 1, 0, n_lat - 1)
    return il * n_lat + jl


def cell_lonlat(cell_id, res):
    n_lon, n_lat = shape(res)
    lc, ltc = centres(res)
    cell_id = np.asarray(cell_id)
    return lc[cell_id // n_lat], ltc[cell_id % n_lat]


def to_grid(cell_id, values, res):
    """(n_lon, n_lat) array, NaN where no value."""
    n_lon, n_lat = shape(res)
    g = np.full(n_lon * n_lat, np.nan)
    g[np.asarray(cell_id)] = np.asarray(values, dtype=float)
    return g.reshape(n_lon, n_lat)


def add_cells(df, cfg):
    df = df.copy()
    df["cell_id"] = cell_index(df.lon.values, df.lat.values, cfg["grid"]["res"])
    return df


def cell_stats(df, col, cfg):
    """Per-cell statistics of ``df[col]`` (see module docstring), indexed by cell_id."""
    g = cfg["grid"]
    folds = cfg["folds"]
    x = df[["cell_id", "fold", col]].rename(columns={col: "x"})
    pooled = x.groupby("cell_id").x.agg(n="size", mean="mean")
    pooled["abs_mean"] = x.assign(a=x.x.abs()).groupby("cell_id").a.mean()
    pf = x.groupby(["cell_id", "fold"]).x.agg(["size", "mean"]).unstack("fold")
    out = pooled.copy()
    means = []
    for f in folds:
        nf = pf["size"][f] if f in pf["size"] else pd.Series(0, index=pf.index)
        mf = pf["mean"][f] if f in pf["mean"] else pd.Series(np.nan, index=pf.index)
        nf = nf.reindex(out.index).fillna(0).astype(int)
        mf = mf.reindex(out.index).where(nf >= int(g["min_per_fold"]))
        out[f"n_f{f}"] = nf
        out[f"f{f}"] = mf
        means.append(mf)
    M = np.stack([m.values for m in means], axis=1)
    out["n_folds"] = np.isfinite(M).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["fold_mean"] = np.nanmean(np.where(np.isfinite(M), M, np.nan), axis=1) \
            if M.size else np.nan
        std = np.nanstd(M, axis=1, ddof=1)
        out["fold_std"] = np.where(out["n_folds"] >= 2, std, np.nan)
        out["ratio"] = out["fold_mean"] / out["fold_std"]
    out["n_pos"] = (M > 0).sum(1)
    out["n_neg"] = (M < 0).sum(1)
    out["sign_consistent"] = np.maximum(out.n_pos, out.n_neg) >= int(g["sign_k"])
    out["ok"] = out.n >= int(g["n_min"])
    lon, lat = cell_lonlat(out.index.values, g["res"])
    out.insert(0, "lat", lat)
    out.insert(0, "lon", lon)
    return out


def tidy(stats, model, quantity, stats_to_keep=("mean", "abs_mean", "fold_mean", "fold_std",
                                                 "ratio")):
    """Long table: cell_id, lon, lat, n, model, quantity, fold, stat, value.
    fold = 'pooled' for the pooled / across-fold statistics, k for the fold-k mean."""
    base = stats.reset_index()[["cell_id", "lon", "lat", "n"]]
    parts = []
    for s in stats_to_keep:
        if s in stats:
            parts.append(base.assign(fold="pooled", stat=s, value=stats[s].values))
    for c in stats.columns:
        if c.startswith("f") and c[1:].isdigit():
            k = int(c[1:])
            parts.append(base.assign(n=stats[f"n_f{k}"].values, fold=k, stat="fold_mean_k",
                                     value=stats[c].values))
    parts.append(base.assign(fold="pooled", stat="sign_consistent",
                             value=stats["sign_consistent"].astype(float).values))
    out = pd.concat(parts, ignore_index=True)
    out.insert(4, "model", model)
    out.insert(5, "quantity", quantity)
    return out


def fold_agg(df, col, by, folds):
    """Per-fold group means of ``col`` then mean / std over folds, per group ``by``."""
    per = df.groupby([by, "fold"])[col].mean().unstack("fold").reindex(columns=folds)
    n = df.groupby(by).size()
    return pd.DataFrame({"n": n, "pooled_mean": df.groupby(by)[col].mean(),
                         "fold_mean": per.mean(1), "fold_std": per.std(1, ddof=1),
                         "n_folds": per.notna().sum(1)})
