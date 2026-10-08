"""House style of every figure: palette, markers, axes, the mean ± std series with hollow
markers where a value is unresolved (|mean/std| < 2 over the folds). Matplotlib mathtext
only (no LaTeX toolchain), headless backend."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DPI = 200
plt.rcParams.update({"text.usetex": False, "mathtext.default": "regular", "font.family": "sans-serif",
                     "savefig.dpi": FIG_DPI, "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

VIEW_ORDER = ["S2_S2VI", "S1", "weather", "DEM"]
VIEW_COLORS = {"S2_S2VI": "#2f7ac4", "S1": "#d97706", "weather": "#b52d20", "DEM": "#0f8a72"}
VIEW_MARKERS = {"S2_S2VI": "o", "S1": "s", "weather": "^", "DEM": "D"}
VIEW_LABELS = {"S2_S2VI": "S2", "S1": "S1", "weather": "weather", "DEM": "DEM"}
INK, INK_MUTED = "#1a1a19", "#6b6b68"
SNR_GATE = 2.0


def style_axes(ax, zero_line=False, labelsize=8):
    if zero_line:
        ax.axhline(0.0, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.grid(alpha=0.25, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=labelsize, length=3)


def draw_series(ax, x, y, s, color, marker, dash_last=False):
    """Line + ±1 std band + markers, hollow where |mean/std| < 2; optional dashed last
    segment (degenerate endpoint)."""
    x, y, s = np.asarray(x, float), np.asarray(y, float), np.asarray(s, float)
    res = np.abs(np.divide(y, s, out=np.full_like(y, np.inf), where=s > 0)) >= SNR_GATE
    ax.fill_between(x, y - s, y + s, color=color, alpha=0.16, lw=0, zorder=2)
    if dash_last and len(x) > 1:
        ax.plot(x[:-1], y[:-1], color=color, lw=2.0, zorder=3, solid_capstyle="round")
        ax.plot(x[-2:], y[-2:], color=color, lw=2.0, ls=(0, (3, 2.5)), zorder=3)
    else:
        ax.plot(x, y, color=color, lw=2.0, zorder=3, solid_capstyle="round")
    ax.scatter(x[res], y[res], marker=marker, s=42, color=color, edgecolors="white", linewidths=1.1, zorder=4)
    ax.scatter(x[~res], y[~res], marker=marker, s=42, facecolors="white", edgecolors=color,
               linewidths=1.6, zorder=4)


def save_fig(fig, out) -> Path:
    """PNG (default) or the suffix of ``out``."""
    out = Path(out)
    if out.suffix == "":
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
