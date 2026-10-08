"""
Maps of the per-sample (local) Shapley values and interactions on a regular lon/lat grid.

Per-sample game (see local_analysis/games.py, value function prob_pred):
    v_x(S) = P(ŷ_full(x) | x_S),   ŷ_full = class predicted with all four sensors,   v_x(∅) = 0.5
so φ > 0 (blue) means the sensor pushes towards what the full model concluded, φ < 0 (red)
against it — agreement with the model's own decision, not accuracy. I_ij < 0 (red):
substitution / redundancy, I_ij > 0 (blue): complementarity. A cell shows the mean over its
samples (each test sample explained once, by the model of its fold); cells with fewer than
``min_count`` samples are blank.
"""
from __future__ import annotations

import itertools

import numpy as np
import matplotlib.pyplot as plt

from . import plotstyle
from .basemap import land

ORDER = plotstyle.VIEW_ORDER
LABELS = plotstyle.VIEW_LABELS
INK, INK_MUTED = plotstyle.INK, plotstyle.INK_MUTED
LAND, LAND_EDGE = "#eceae5", "#c9c6bf"
PAIRS = list(itertools.combinations(ORDER, 2))


def short(v):
    return "S2" if v == "S2_S2VI" else v


def phi_col(v):
    return f"phi_{short(v)}"


def sii_col(a, b):
    return f"I_{short(a)}x{short(b)}"


def grid_mean(lons, lats, values, res=2.0, min_count=3):
    """Cell mean on a regular res-degree grid; NaN below min_count."""
    lon_edges = np.arange(-180, 180 + res, res)
    lat_edges = np.arange(-90, 90 + res, res)
    n_lon, n_lat = len(lon_edges) - 1, len(lat_edges) - 1
    lon_i = np.clip(np.digitize(lons, lon_edges) - 1, 0, n_lon - 1)
    lat_i = np.clip(np.digitize(lats, lat_edges) - 1, 0, n_lat - 1)
    flat = lon_i * n_lat + lat_i
    ok = ~np.isnan(values)
    tot = np.bincount(flat[ok], weights=values[ok], minlength=n_lon * n_lat)
    cnt = np.bincount(flat[ok], minlength=n_lon * n_lat).astype(float)
    mean = np.where(cnt >= min_count, tot / np.maximum(cnt, 1), np.nan)
    return (mean.reshape(n_lon, n_lat), cnt.reshape(n_lon, n_lat),
            (lon_edges[:-1] + lon_edges[1:]) / 2, (lat_edges[:-1] + lat_edges[1:]) / 2)


def grid_std(lons, lats, values, res=2.0, min_count=3):
    """Within-cell standard deviation (ddof=1); NaN below min_count."""
    mean, cnt, lc, ltc = grid_mean(lons, lats, values, res, min_count=1)
    sq, _, _, _ = grid_mean(lons, lats, values ** 2, res, min_count=1)
    var = (sq - mean ** 2) * cnt / np.maximum(cnt - 1, 1)
    std = np.where(cnt >= max(min_count, 2), np.sqrt(np.clip(var, 0, None)), np.nan)
    return std, cnt, lc, ltc


def setup(ax, background, lat_lim):
    if background is not None:
        background.plot(ax=ax, color=LAND, edgecolor=LAND_EDGE, linewidth=0.35, zorder=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(*lat_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_color(LAND_EDGE)
        s.set_linewidth(0.6)


def grids(df, cols, res=2.0, min_count=3, stat="mean"):
    """{col: (values, lon_centres, lat_centres)}"""
    fn = grid_std if stat == "std" else grid_mean
    out = {}
    for c in cols:
        vals, _, lc, ltc = fn(df["lon"].values, df["lat"].values, df[c].values, res, min_count)
        out[c] = (vals, lc, ltc)
    return out


def shared_vmax(grid_sets, q=98):
    """q-th percentile of |cell value| pooled over every panel of every grid set."""
    return float(np.nanpercentile(np.abs(np.concatenate(
        [g[0][np.isfinite(g[0])].ravel() for gs in grid_sets for g in gs.values()])), q))


def _panels(df, cols, titles, out_path, *, nrows, ncols, figsize, adjust, cax_rect, cbar_label,
            title, stat, vmax, res, min_count, lat_lim):
    is_std = stat == "std"
    gs = grids(df, cols, res, min_count, stat)
    vmax = shared_vmax([gs]) if vmax is None else vmax
    bg = land()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.subplots_adjust(**adjust)
    pc = None
    for ax, c, t in zip(axes.flatten(), cols, titles):
        vals, lc, ltc = gs[c]
        setup(ax, bg, lat_lim)
        pc = ax.pcolormesh(lc, ltc, vals.T, cmap="viridis" if is_std else "RdBu",
                           vmin=0.0 if is_std else -vmax, vmax=vmax, shading="auto", zorder=2)
        tag = f"std {df[c].std():.3f}" if is_std else f"mean {df[c].mean():+.3f}"
        ax.set_title(f"{t}      {tag}", fontsize=9.5, color=INK, loc="left", pad=4)
    cb = fig.colorbar(pc, cax=fig.add_axes(cax_rect), orientation="horizontal")
    cb.set_label(cbar_label, fontsize=8.5, color=INK)
    cb.ax.tick_params(labelsize=7.5, colors=INK_MUTED, length=3)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(LAND_EDGE)
    fig.suptitle(title, fontsize=11.5, color=INK, x=adjust["left"], ha="left", y=0.985)
    return plotstyle.save_fig(fig, out_path)


def fig_local_shapley_maps(df, out_path, model_label=None, res=2.0, min_count=3, lat_lim=(-58, 68),
                           stat="mean", vmax=None):
    """2x2 map of the cell mean (or within-cell std) of the local phi, one shared colour bar."""
    is_std = stat == "std"
    title = f"Local Shapley values, {'standard deviation' if is_std else 'mean'} per grid cell"
    return _panels(df, [phi_col(v) for v in ORDER], [LABELS[v] for v in ORDER], out_path,
                   nrows=2, ncols=2, figsize=(11.4, 5.5),
                   adjust=dict(left=0.012, right=0.988, top=0.885, bottom=0.145, wspace=0.045, hspace=0.20),
                   cax_rect=[0.335, 0.105, 0.33, 0.022],
                   cbar_label=(r"std of local $\phi$ within the cell" if is_std else
                               r"mean local $\phi$, blue support the full-coalition prediction, red goes against it"),
                   title=title + (f" - {model_label}" if model_label else ""), stat=stat, vmax=vmax,
                   res=res, min_count=min_count, lat_lim=lat_lim)


def fig_local_interaction_maps(df, out_path, model_label=None, res=2.0, min_count=3, lat_lim=(-58, 68),
                               stat="mean", vmax=None):
    """2x3 map of the cell mean (or within-cell std) of the local pairwise interaction index."""
    is_std = stat == "std"
    title = f"Local Shapley interactions, {'standard deviation' if is_std else 'mean'} per grid cell"
    return _panels(df, [sii_col(a, b) for a, b in PAIRS], [f"{LABELS[a]} × {LABELS[b]}" for a, b in PAIRS],
                   out_path, nrows=2, ncols=3, figsize=(15.6, 4.9),
                   adjust=dict(left=0.008, right=0.992, top=0.875, bottom=0.14, wspace=0.035, hspace=0.20),
                   cax_rect=[0.35, 0.085, 0.30, 0.024],
                   cbar_label=(r"std of local interaction $\phi_{ij}$ within the cell" if is_std else
                               r"mean local interaction $\phi_{ij}$, blue complementary, red redundant"),
                   title=title + (f" - {model_label}" if model_label else ""), stat=stat, vmax=vmax,
                   res=res, min_count=min_count, lat_lim=lat_lim)
