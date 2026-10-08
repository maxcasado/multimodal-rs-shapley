"""
Order-2 Shapley interactions of the GLOBAL f1_weighted game, per fold, model and level, for
the main experiment and both S2 ablations, with v(∅) = 0.5127:

    I_ij   = Σ_{S⊆N∖{i,j}} |S|!(n-|S|-2)!/(n-1)! Δ_ij(S),  n = 4 (weights 1/3, 1/6, 1/6, 1/3)
    I_vide = (1/3) Δ_ij(∅) = (1/3)[v({i,j}) - v({i}) - v({j}) + v(∅)],   I_ctx = I - I_vide
    d(T)   = Σ_{S⊆T} (-1)^{|T|-|S|} v(S)  for the 15 non-empty T (Harsanyi dividends)

aggregated over the folds (mean, std ddof=1, SNR, unresolved when |SNR| < 2). Check (hard):
I_ij = Σ_{T⊇{i,j}} d(T)/(|T|-1) and φ_i = Σ_{T∋i} d(T)/|T| on every game.

    data/interactions_global{,_per_fold,_harsanyi}.csv, data/interactions_global_checks.json
    figures/interactions_degradation.pdf   rows noise | cloud gap, columns CoM | EmbraceNet
"""
import json

import pandas as pd

from shapfusion import config as C
from shapfusion.analysis import games as G
from shapfusion.analysis.stats import SNR_THRESHOLD, summary

QTY = ("I", "I_vide", "I_ctx")
PAIR_COLORS = ["#0961bb", "#ac3225", "#bd8630", "#6d8af3", "#c673a3", "#70306c"]
PAIR_MARKERS = ["o", "s", "^", "D", "v", "P"]


def level_label(cfg, experiment: str, level: float) -> str:
    if experiment == "noise":
        return f"sigma={level:g}"
    if experiment == "cloud_gap":
        n = G.seq_len(cfg)
        return f"{round(level * n)}/{n} ({level:.0%})"
    return "-"


def per_fold(games: pd.DataFrame):
    rows, drows, err = [], [], {"I": 0.0, "phi": 0.0}
    for r in games.itertuples(index=False):
        key = {"model": r.model, "experiment": r.experiment, "level": r.level, "fold": r.fold}
        for pair, (i, iv, ic) in G.interaction_split(r.game).items():
            rows.append({**key, "pair": G.pair_label(pair), "I": i, "I_vide": iv, "I_ctx": ic})
        for T, d in G.harsanyi(r.game).items():
            drows.append({**key, "T": "+".join(G.VIEW_LABELS[v] for v in G.VIEW_ORDER if v in T),
                          "size": len(T), "d": d})
        e = G.identity_errors(r.game)
        err = {k: max(err[k], e[k]) for k in err}
    return pd.DataFrame(rows), pd.DataFrame(drows), err


def summarise(cfg, pf):
    s = summary(pf, ["model", "experiment", "level", "pair"], QTY)
    s.insert(3, "level_label", [level_label(cfg, e, l) for e, l in zip(s["experiment"], s["level"])])
    return s


def figure(cfg, s, out):
    from shapfusion.analysis import plotstyle as ps
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    models = G.models(cfg)
    fig, axes = plt.subplots(2, len(models), figsize=(12.0, 8.6), sharey="row", squeeze=False)
    for col, model in enumerate(models):
        for row, exp in enumerate(("noise", "cloud_gap")):
            ax = axes[row, col]
            d = s[(s["model"] == model) & (s["experiment"] == exp)]
            levels = sorted(d["level"].unique())
            for pair, color, marker in zip(G.PAIRS, PAIR_COLORS, PAIR_MARKERS):
                sub = d[d["pair"] == G.pair_label(pair)].sort_values("level")
                x = [levels.index(l) for l in sub["level"]] if exp == "noise" else sub["level"].values
                ps.draw_series(ax, x, sub["I_mean"], sub["I_std"], color, marker, dash_last=(exp == "cloud_gap"))
            ps.style_axes(ax, zero_line=True)
            ax.tick_params(labelleft=True)
            if exp == "noise":
                ax.set_xticks(range(len(levels)))
                ax.set_xticklabels([f"{x:g}" for x in levels], fontsize=8)
                ax.set_xlim(-0.25, len(levels) - 0.75)
                ax.set_xlabel(r"Gaussian noise $\sigma$ on S2", fontsize=9.5, color=ps.INK)
            else:
                ticks = G.gap_levels(cfg)
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{t:.0%}" for t in ticks], fontsize=7)
                ax.set_xlim(-0.03, 1.03)
                ax.set_xlabel(f"share of the {G.seq_len(cfg)} S2 dates lost and linearly interpolated",
                              fontsize=9.5, color=ps.INK)
            if col == 0:
                ax.set_ylabel(r"shapley interaction $I_{ij}$", fontsize=9.5, color=ps.INK)
            if row == 0:
                ax.set_title(model, fontsize=11, color=ps.INK, loc="left", pad=8)
    handles = [Line2D([], [], color=c, marker=m, lw=2.0, ms=6, markeredgecolor="white", markeredgewidth=1.0,
                      label=G.pair_label(p).replace("-", " – ")) for p, c, m in zip(G.PAIRS, PAIR_COLORS, PAIR_MARKERS)]
    handles.append(Line2D([], [], color=ps.INK_MUTED, marker="o", lw=0, ms=6, markerfacecolor="white",
                          markeredgecolor=ps.INK_MUTED, markeredgewidth=1.4,
                          label=f"unresolved (|SNR| < {SNR_THRESHOLD:g})"))
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.035), fontsize=8.5,
                     ncol=len(handles), frameon=True, framealpha=0.92, edgecolor="#dcdcd8")
    leg.get_frame().set_linewidth(0.6)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.075, wspace=0.10, hspace=0.24)
    ps.save_fig(fig, out)


def run(cfg):
    games = G.load_games(cfg)
    pf, dv, err = per_fold(games)
    if max(err.values()) > 1e-10:
        raise AssertionError(f"Harsanyi identities violated: {err}")
    s = summarise(cfg, pf)
    harsanyi = summary(dv, ["model", "experiment", "level", "T", "size"], ["d"])
    checks = {"v_empty": G.V_EMPTY_F1, "n_games": len(games), "identity_max_abs_err": err}
    data = C.out(cfg, "data")
    s.to_csv(data / "interactions_global.csv", index=False)
    pf.to_csv(data / "interactions_global_per_fold.csv", index=False)
    harsanyi.to_csv(data / "interactions_global_harsanyi.csv", index=False)
    (data / "interactions_global_checks.json").write_text(json.dumps(checks, indent=2))
    figure(cfg, s, C.out(cfg, "figures") / "interactions_degradation.pdf")
    print(f"    identities over {len(games)} games: I {err['I']:.1e}, phi {err['phi']:.1e}")
