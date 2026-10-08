"""
World-grid panel figures for the local analyses: rows (usually models) × columns
(sensors / pairs / components), ONE colour scale per figure.

Conventions (every figure):
  * colour limit vmax = q-th percentile (cfg colour.vmax_q, 99) of |cell value| pooled
    over every panel of the figure, i.e. over all the models it shows; the share of
    cells beyond it (clipped) is printed in each panel title and returned;
  * cells with 0 < n < n_min are HATCHED, never coloured;
  * cells masked for another reason (e.g. difference without a consistent sign) are
    drawn plain light grey; optional black dots mark cells whose sign is not consistent;
  * the legend line under the colour bar states the value function, n_min and vmax.
RdBu for signed maps (blue = supports / complementary, red = against / redundant),
viridis for unsigned ones — as shapfusion/analysis/local_maps.py.
"""
import numpy as np
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt

from shapfusion.analysis import plotstyle
from shapfusion.analysis.basemap import land
from shapfusion.analysis.local_maps import INK, INK_MUTED, LAND_EDGE, setup as _setup
from . import grid

HATCH_EDGE = "#8c8c8c"
MASK_FACE = "#d4d4d4"
_BORDERS = []


def borders():
    if not _BORDERS:
        _BORDERS.append(land())
    return _BORDERS[0]


def colour_limit(panels, q, signed=True):
    vals = [np.abs(p["values"]) if signed else p["values"] for row in panels for p in row
            if p is not None]
    vals = np.concatenate([v[np.isfinite(v)].ravel() for v in vals]) if vals else np.array([])
    return float(np.nanpercentile(vals, q)) if vals.size else 1.0


def clipped_pct(values, vmin, vmax):
    v = values[np.isfinite(values)]
    return float(100.0 * ((v > vmax) | (v < vmin)).mean()) if v.size else float("nan")


def panel(stats, col, cfg, mask=None, stipple=None):
    """Panel dict from a cell_stats frame: values of ``col`` where n >= n_min (and not
    ``mask``), hatch where 0 < n < n_min."""
    res = cfg["grid"]["res"]
    ok = stats.ok.values
    v = stats[col].values.astype(float).copy()
    v[~ok] = np.nan
    masked = np.zeros(len(stats), bool) if mask is None else (np.asarray(mask) & ok)
    v[masked] = np.nan
    p = {"values": grid.to_grid(stats.index.values, v, res),
         "low": grid.to_grid(stats.index.values, (~ok).astype(float), res) == 1,
         "masked": grid.to_grid(stats.index.values, masked.astype(float), res) == 1}
    if stipple is not None:
        st = np.asarray(stipple) & ok & ~masked & np.isfinite(v)
        p["stipple"] = grid.to_grid(stats.index.values, st.astype(float), res) == 1
    return p


def figure(panels, out_path, cfg, *, row_labels, col_titles, title, cbar_label, legend,
           signed=True, vmax=None, vmin=None, log=False, cmap=None, lat_lim=(-58, 68),
           groups=None):
    """Draw ``panels`` (list of rows of panel dicts / None).

    One colour scale per column GROUP (default: one group = every column). ``groups`` is a
    list of dicts {cols, signed, vmax, vmin, log, cmap, cbar_label}; missing keys fall
    back to the figure-level arguments. vmax of a group = q-th percentile of |value| over
    every panel of its columns (all rows / models) unless given.
    Returns {"vmax", "vmin", "clipped": [[pct, ...], ...], "path", "groups"}."""
    res = cfg["grid"]["res"]
    q = cfg["colour"]["vmax_q"]
    nr, nc = len(panels), max(len(r) for r in panels)
    groups = groups or [{"cols": list(range(nc))}]
    col_norm, col_cmap, gout = {}, {}, []
    for g in groups:
        sg = g.get("signed", signed)
        lg = g.get("log", log)
        sub = [[row[c] if c < len(row) else None for c in g["cols"]] for row in panels]
        vx = g.get("vmax", vmax if len(groups) == 1 else None)
        if vx is None:
            vx = colour_limit(sub, q, sg)
        vn = g.get("vmin", vmin if len(groups) == 1 else None)
        if vn is None:
            vn = -vx if sg else 0.0
        cm = plt.get_cmap(g.get("cmap", cmap) or ("RdBu" if sg else "viridis")).copy()
        nm = (mcolors.LogNorm(vmin=max(vn, 1), vmax=vx) if lg
              else mcolors.Normalize(vmin=vn, vmax=vx))
        for c in g["cols"]:
            col_norm[c], col_cmap[c] = nm, cm
        gout.append({"cols": g["cols"], "vmax": float(nm.vmax), "vmin": float(nm.vmin),
                     "signed": sg, "log": lg, "cmap": cm,
                     "cbar_label": g.get("cbar_label", cbar_label), "mappable": None})
    lc, ltc = grid.centres(res)
    lon_e, lat_e = grid.edges(res)
    pw = 3.35
    ph = pw * (lat_lim[1] - lat_lim[0]) / 360.0 + 0.36
    fig_w, fig_h = 0.55 + pw * nc, 0.55 + ph * nr + 1.05
    fig, axes = plt.subplots(nr, nc, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.55 / fig_w, right=0.995, top=1 - 0.5 / fig_h,
                        bottom=1.05 / fig_h, wspace=0.03, hspace=0.30)
    clipped = []
    b = borders()
    for r, row in enumerate(panels):
        crow = []
        for c in range(nc):
            ax = axes[r, c]
            p = row[c] if c < len(row) else None
            if p is None:
                ax.set_axis_off()
                crow.append(np.nan)
                continue
            _setup(ax, b, lat_lim)
            if p.get("masked") is not None and p["masked"].any():
                m = np.where(p["masked"], 1.0, np.nan)
                ax.pcolormesh(lon_e, lat_e, m.T, cmap=mcolors.ListedColormap([MASK_FACE]),
                              vmin=0, vmax=1, shading="flat", zorder=2)
            if p.get("low") is not None and p["low"].any():
                m = np.ma.masked_invalid(np.where(p["low"], 1.0, np.nan))
                h = ax.pcolor(lon_e, lat_e, m.T, cmap=mcolors.ListedColormap(["none"]),
                              edgecolors=HATCH_EDGE, linewidth=0.0, zorder=2)
                h.set_hatch("//////")
                h.set_facecolor("none")
            norm = col_norm[c]
            pc = ax.pcolormesh(lon_e, lat_e, p["values"].T, cmap=col_cmap[c], norm=norm,
                               shading="flat", zorder=3)
            for g in gout:
                if c in g["cols"] and g["mappable"] is None:
                    g["mappable"] = pc
            if p.get("stipple") is not None and p["stipple"].any():
                ii, jj = np.where(p["stipple"])
                ax.scatter(lc[ii], ltc[jj], s=0.6, c=INK, linewidths=0, zorder=4)
            pct = clipped_pct(p["values"], norm.vmin, norm.vmax)
            crow.append(pct)
            t = p.get("title", col_titles[c] if c < len(col_titles) else "")
            ax.set_title(f"{t}   clip {pct:.1f}%".strip(), fontsize=8.2, color=INK,
                         loc="left", pad=2.5)
            if c == 0:
                ax.text(-0.03, 0.5, row_labels[r], transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=8.6, color=INK)
        clipped.append(crow)
    fig.suptitle(title, fontsize=11, color=INK, x=0.55 / fig_w, ha="left",
                 y=1 - 0.12 / fig_h)
    for g in gout:
        if g["mappable"] is None:
            continue
        x0 = axes[0, min(g["cols"])].get_position().x0
        x1 = axes[0, max(g["cols"])].get_position().x1
        w = min(0.32 if len(gout) == 1 else 0.8 * (x1 - x0), x1 - x0)
        cax = fig.add_axes([(x0 + x1) / 2 - w / 2, 0.66 / fig_h, w, 0.13 / fig_h])
        cb = fig.colorbar(g["mappable"], cax=cax, orientation="horizontal",
                          extend="neither" if g["log"] else ("both" if g["signed"] else "max"))
        cb.set_label(g["cbar_label"], fontsize=8.2, color=INK)
        cb.ax.tick_params(labelsize=7.2, colors=INK_MUTED, length=3)
        cb.outline.set_linewidth(0.5)
        cb.outline.set_edgecolor(LAND_EDGE)
    # legend hangs from just under the colour-bar label, so a second line grows downwards
    fig.text(0.5, 0.26 / fig_h, legend, ha="center", va="top", fontsize=7.6, color=INK,
             linespacing=1.35)
    path = plotstyle.save_fig(fig, out_path)
    for g in gout:
        g.pop("mappable")
        g.pop("cmap")
    return {"vmax": gout[0]["vmax"], "vmin": gout[0]["vmin"], "clipped": clipped,
            "path": str(path), "groups": gout}


def legend_text(cfg, value_function, vmax, extra=""):
    g = cfg["grid"]
    s = (f"{value_function}   |   {g['res']:g}° cells, n_min = {g['n_min']} "
         f"(hatched: 0 < n < n_min)   |   vmax = {vmax:.3g} "
         f"(p{cfg['colour']['vmax_q']:g} of |cell value| over all panels)")
    s += f"   |   {extra}" if extra else ""
    if cfg.get("_quant_note"):
        s += "\n" + cfg["_quant_note"]
    return s


def scatter_grid(xs, ys, out_path, *, row_labels, col_titles, xlabel, ylabel, title,
                 weights=None, annotate=None, legend=""):
    """Rows × cols of scatter plots (cells), with an annotation string per panel."""
    nr, nc = len(xs), len(xs[0])
    fig, axes = plt.subplots(nr, nc, figsize=(0.6 + 3.0 * nc, 0.9 + 2.5 * nr), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.12, wspace=0.28, hspace=0.42)
    for r in range(nr):
        for c in range(nc):
            ax = axes[r, c]
            x, y = np.asarray(xs[r][c]), np.asarray(ys[r][c])
            s = 4 if weights is None else 1 + 10 * np.sqrt(np.asarray(weights[r][c]) /
                                                             np.max(weights[r][c]))
            ax.scatter(x, y, s=s, c=plotstyle.VIEW_COLORS["S2_S2VI"], alpha=0.35,
                       linewidths=0)
            plotstyle.style_axes(ax, zero_line=True, labelsize=7)
            ax.axvline(0, color=INK_MUTED, lw=0.6, ls=(0, (4, 3)))
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8.5, color=INK)
            if c == 0:
                ax.set_ylabel(f"{row_labels[r]}\n{ylabel}", fontsize=8, color=INK)
            if r == nr - 1:
                ax.set_xlabel(xlabel if isinstance(xlabel, str) else xlabel[c], fontsize=8,
                              color=INK)
            if annotate is not None:
                ax.text(0.02, 0.97, annotate[r][c], transform=ax.transAxes, va="top",
                        fontsize=7.2, color=INK,
                        bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=1.5))
    fig.suptitle(title, fontsize=10.5, color=INK, x=0.01, ha="left")
    if legend:
        fig.text(0.5, 0.01, legend, ha="center", va="bottom", fontsize=7.4, color=INK)
    return str(plotstyle.save_fig(fig, out_path))
