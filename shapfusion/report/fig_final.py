"""
The paper's main figures, CoM | EmbraceNet side by side (shared y scale), bands = ±1 std
over the folds, HOLLOW marker = unresolved (|mean/std| < 2):

  fig_noise_shapley.png          phi vs Gaussian noise sigma on S2 (equally spaced sigma ticks)
  fig_noise_ps.png               Perceptual Score, same sweep
  fig_phi_decomposition.png      phi = phi_grand + phi_small (main experiment)
  fig_cloud_gap_shapley.png      phi vs share of the 12 S2 dates lost and re-interpolated
  fig_cloud_gap_f1.png           full-coalition f1_weighted, same sweep
  fig_shapley_local_map_<model>.png   local phi, mean per 2-degree cell (4 panels)

and the data behind them in data/: noise_shapley_{summary,per_fold}.csv,
decomposition_summary.csv, cloud_gap_shapley_summary.csv, cloud_gap_f1_summary.csv.
Every phi uses the paper's v(∅) = 0.5127 (analysis/games.py).
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from shapfusion import config as C
from shapfusion.analysis import games as G, local_maps, plotstyle
from shapfusion.analysis.stats import summary
from shapfusion.local_analysis import config as la_config, games as LG

ORDER = plotstyle.VIEW_ORDER
COLORS, MARKERS, LABELS = plotstyle.VIEW_COLORS, plotstyle.VIEW_MARKERS, plotstyle.VIEW_LABELS
INK, INK_MUTED = plotstyle.INK, plotstyle.INK_MUTED
SNR = 2.0
NOISE_TOL = 0.01          # phi at sigma = 0 (a retrained model) vs the main experiment


def _summary(pf, keys, cols):
    return summary(pf, keys, cols, inf_resolved=True)


def _legend(ax):
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], lw=2.0, ms=6, markeredgecolor="white",
                      markeredgewidth=1.0, label=LABELS[m]) for m in ORDER]
    handles.append(Line2D([], [], color=INK_MUTED, marker="o", lw=0, ms=6, markerfacecolor="white",
                          markeredgecolor=INK_MUTED, markeredgewidth=1.4, label=f"unresolved (|SNR| < {SNR:g})"))
    leg = ax.legend(handles=handles, loc="best", fontsize=8, frameon=True, framealpha=0.92, edgecolor="#dcdcd8")
    leg.get_frame().set_linewidth(0.6)
    for t in leg.get_texts():
        t.set_color(INK)


def _panels(n=2, figsize=(11.0, 4.3)):
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.tick_params(labelleft=True)
    return fig, axes


def _title(ax, name):
    ax.set_title(name, fontsize=11, color=INK, loc="left", pad=8)


# ── noise sweep ──────────────────────────────────────────────────────────────
def _noise_panel(ax, df, value, err):
    sigmas = sorted(df["sigma"].unique())
    pos = {s: i for i, s in enumerate(sigmas)}
    for m in ORDER:
        sub = df[df["modality"] == m].sort_values("sigma")
        plotstyle.draw_series(ax, [pos[s] for s in sub["sigma"]], sub[value], sub[err], COLORS[m], MARKERS[m])
    plotstyle.style_axes(ax, zero_line=True)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"{s:g}" for s in sigmas], fontsize=8)
    ax.set_xlim(-0.25, len(sigmas) - 0.75)
    ax.set_xlabel(r"Gaussian noise $\sigma$ on S2", fontsize=9.5, color=INK)


def noise_phi_summary(cfg, checks):
    pf = G.phi_frame(G.load_games(cfg, experiments=("main", "noise")))
    out = _summary(pf, ["model", "experiment", "level", "modality"], ["phi"])
    main = out[out["experiment"] == "main"]
    out = out[out["experiment"] == "noise"].drop(columns="experiment").rename(columns={"level": "sigma"})
    out["v_empty"] = G.V_EMPTY_F1
    data = C.out(cfg, "data")
    out.to_csv(data / "noise_shapley_summary.csv", index=False)
    pf[pf["experiment"] == "noise"].to_csv(data / "noise_shapley_per_fold.csv", index=False)
    for model in G.models(cfg):
        ref = main[main["model"] == model].set_index("modality")["phi_mean"]
        s0 = out[(out["model"] == model) & (out["sigma"] == 0.0)].set_index("modality")["phi_mean"]
        diff = {LABELS[m]: round(float(s0[m] - ref[m]), 4) for m in ORDER}
        ok = all(abs(v) <= NOISE_TOL for v in diff.values())
        checks[f"noise_sigma0_vs_main_{model}"] = {"phi_diff": diff, "tolerance": NOISE_TOL, "pass": ok}
        print(f"    sigma=0 vs main, {model}: phi diff {diff} -> {'PASS' if ok else 'FAIL'} (±{NOISE_TOL})")
    return out


def noise_figures(cfg, checks):
    phi = noise_phi_summary(cfg, checks)
    data = {}
    for n in G.models(cfg):
        ps = _summary(G.load_ps(cfg, n, "noise"), ["level", "modality"], ["PS_raw"]).rename(
            columns={"level": "sigma", "PS_raw_mean": "ps_mean", "PS_raw_std": "ps_std"})
        data[n] = ps[["sigma", "modality", "ps_mean", "ps_std"]].merge(
            phi[phi["model"] == n].drop(columns="model"), on=["sigma", "modality"])
    figs = C.out(cfg, "figures")
    for fname, value, err, ylab in [("fig_noise_shapley.png", "phi_mean", "phi_std", r"shapley value $\phi$"),
                                    ("fig_noise_ps.png", "ps_mean", "ps_std", "Perceptual Score")]:
        fig, axes = _panels(figsize=(11.0, 4.4))
        for ax, name in zip(axes, G.models(cfg)):
            _noise_panel(ax, data[name], value, err)
            _title(ax, name)
        axes[0].set_ylabel(ylab, fontsize=9.5, color=INK)
        _legend(axes[0])
        fig.subplots_adjust(left=0.065, right=0.99, top=0.91, bottom=0.135, wspace=0.135)
        plotstyle.save_fig(fig, figs / fname)


# ── phi = phi_grand + phi_small ──────────────────────────────────────────────
def decomposition_figure(cfg):
    dec = G.decomposition_frame(G.load_games(cfg, experiments=("main",)))
    _summary(dec, ["model", "modality"], ["phi", "phi_grand", "phi_small"]).to_csv(
        C.out(cfg, "data") / "decomposition_summary.csv", index=False)
    fig, axes = _panels(figsize=(9.4, 4.3))
    for ax, name in zip(axes, G.models(cfg)):
        g = dec[dec["model"] == name].groupby("modality").agg(
            grand=("phi_grand", "mean"), grand_s=("phi_grand", "std"), small=("phi_small", "mean"),
            phi_s=("phi", "std"))
        x = np.arange(len(ORDER))
        grand = np.array([g.loc[m, "grand"] for m in ORDER])
        small = np.array([g.loc[m, "small"] for m in ORDER])
        ax.bar(x, grand, width=0.62, color=[COLORS[m] for m in ORDER], zorder=3)
        ax.bar(x, small, width=0.62, bottom=grand, color=[COLORS[m] for m in ORDER], alpha=0.35, zorder=3)
        ax.errorbar(x, grand, yerr=[g.loc[m, "grand_s"] for m in ORDER], fmt="none", ecolor=INK,
                    elinewidth=0.9, capsize=2.5, zorder=4)
        ax.errorbar(x, grand + small, yerr=[g.loc[m, "phi_s"] for m in ORDER], fmt="none", ecolor=INK,
                    elinewidth=0.9, capsize=2.5, zorder=4)
        plotstyle.style_axes(ax, zero_line=True)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in ORDER], fontsize=9, color=INK)
        ax.tick_params(axis="x", length=0)
        _title(ax, name)
    axes[0].set_ylabel("shapley value decomposition", fontsize=9.5, color=INK)
    leg = axes[-1].legend(handles=[Patch(facecolor=INK_MUTED, label=r"$\phi^{\mathrm{grand}}$"),
                                   Patch(facecolor=INK_MUTED, alpha=0.35, label=r"$\phi^{\mathrm{small}}$")],
                          loc="upper right", fontsize=9, frameon=True, framealpha=0.92, edgecolor="#dcdcd8")
    leg.get_frame().set_linewidth(0.6)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.91, bottom=0.10, wspace=0.115)
    plotstyle.save_fig(fig, C.out(cfg, "figures") / "fig_phi_decomposition.png")


# ── cloud-gap sweep ──────────────────────────────────────────────────────────
def _gap_axis(ax, cfg):
    plotstyle.style_axes(ax)
    ticks = G.gap_levels(cfg)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks], fontsize=7)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(f"share of the {G.seq_len(cfg)} S2 dates lost and linearly interpolated", fontsize=9.5, color=INK)


def cloud_gap_figures(cfg):
    games = G.load_games(cfg, experiments=("cloud_gap",))
    gap = _summary(G.phi_frame(games), ["model", "level", "modality"], ["phi"]).rename(columns={"level": "fraction"})
    gap["v_empty"] = G.V_EMPTY_F1
    gap.to_csv(C.out(cfg, "data") / "cloud_gap_shapley_summary.csv", index=False)
    figs = C.out(cfg, "figures")
    fig, axes = _panels(figsize=(12.0, 4.6))
    for ax, name in zip(axes, G.models(cfg)):
        d = gap[gap["model"] == name]
        for m in ORDER:
            sub = d[d["modality"] == m].sort_values("fraction")
            plotstyle.draw_series(ax, sub["fraction"], sub["phi_mean"], sub["phi_std"], COLORS[m], MARKERS[m],
                                  dash_last=True)
        _gap_axis(ax, cfg)
        ax.axhline(0.0, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        _title(ax, name)
    axes[0].set_ylabel(r"shapley value $\phi$", fontsize=9.5, color=INK)
    _legend(axes[0])
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.145, wspace=0.10)
    plotstyle.save_fig(fig, figs / "fig_cloud_gap_shapley.png")

    full = frozenset(ORDER)
    vfull = games.assign(v_full=[g[full] for g in games["game"]])
    f1 = _summary(vfull, ["model", "level"], ["v_full"]).rename(columns={"level": "fraction"})
    f1.to_csv(C.out(cfg, "data") / "cloud_gap_f1_summary.csv", index=False)
    fig, axes = _panels(figsize=(12.0, 4.0))
    for ax, name in zip(axes, G.models(cfg)):
        f = f1[f1["model"] == name].sort_values("fraction")
        x, y, s = f["fraction"].values, f["v_full_mean"].values, f["v_full_std"].values
        ax.fill_between(x, y - s, y + s, color=INK, alpha=0.12, lw=0, zorder=2)
        ax.plot(x[:-1], y[:-1], color=INK, lw=2.0, zorder=3, solid_capstyle="round")
        ax.plot(x[-2:], y[-2:], color=INK, lw=2.0, ls=(0, (3, 2.5)), zorder=3)
        ax.scatter(x, y, marker="o", s=42, color=INK, edgecolors="white", linewidths=1.1, zorder=4)
        _gap_axis(ax, cfg)
        _title(ax, name)
    axes[0].set_ylabel("f1_weighted", fontsize=9.5, color=INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.165, wspace=0.10)
    plotstyle.save_fig(fig, figs / "fig_cloud_gap_f1.png")


# ── local Shapley maps ───────────────────────────────────────────────────────
def local_maps_figures(cfg):
    la = la_config.build(cfg)
    for name in G.models(cfg):
        df = LG.sample_frame(la, name, "prob_pred", with_regions=False)
        local_maps.fig_local_shapley_maps(df, C.out(cfg, "figures") / f"fig_shapley_local_map_{name}.png",
                                          model_label=name)


def run(cfg):
    checks = {}
    print("  noise");          noise_figures(cfg, checks)
    print("  decomposition");  decomposition_figure(cfg)
    print("  cloud gap");      cloud_gap_figures(cfg)
    print("  local maps");     local_maps_figures(cfg)
    (C.out(cfg, "data") / "checks_main_figures.json").write_text(json.dumps(checks, indent=2))
