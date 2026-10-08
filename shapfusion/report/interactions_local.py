"""
Order-2 Shapley interactions of the LOCAL (per-sample) game of the local maps, CoM and
EmbraceNet, every test sample explained once by the model of its fold:

    v_x(S) = P(ŷ_full(x) | x_S),  v_x(∅) = 0.5  (EmbraceNet: mean over the R draws)
    I_ij(x) = Σ_{S⊆N∖{i,j}} w(|S|) Δ_ij(S),  w = 1/3, 1/6, 1/6, 1/3;   I_ctx = I - (1/3) Δ_ij(∅)

I < 0: substitution (red), I > 0: complementarity (blue). Check (hard) on 100 random samples
per model: I_ij(x) = Σ_{T⊇{i,j}} d_x(T)/(|T|-1) (Harsanyi) and Σ_i φ_i(x) = v_x(N) - 0.5.

    data/interactions_local_<model>.csv.gz     one row per sample: phi_<view>, I_<pair>, I_ctx_<pair>
    data/interactions_local_summary.csv        per model, pair, subset: mean / median I, share I < 0, ...
    data/interactions_local_cells_s2_weather.csv   every 2° cell (>= 3 samples), S2-weather
    data/interactions_local_checks.json, numbers/interactions_local.md
    figures/interactions_local_s2_weather.pdf  I(S2, weather), one row per model
    figures/interactions_local_pairs.pdf       the 6 pairs of the first model
"""
import json

import numpy as np
import pandas as pd

from shapfusion import config as C
from shapfusion.analysis import games as G
from shapfusion.analysis.local_maps import LAND_EDGE, grid_mean, setup
from shapfusion.analysis.basemap import land
from shapfusion.local_analysis import config as la_config, games as LG, vcache
from shapfusion.shapley import core as sh

V = G.VIEW_ORDER
LAB = G.VIEW_LABELS
VF = "prob_pred"
V_EMPTY = 0.5
RES, MIN_COUNT, VMAX_Q = 2.0, 3, 98
LAT_LIM = (-58, 68)
N_CHECK, SEED, TOL = 100, 0, 1e-10
S2W = ("S2_S2VI", "weather")
SUBSETS = [("all", None), ("cultivated", ("label", 1)), ("non-cultivated", ("label", 0)),
           ("well classified", ("correct", 1)), ("misclassified", ("correct", 0))]


def col(pair, q="I") -> str:
    return f"{q}_{LAB[pair[0]]}_{LAB[pair[1]]}"


def sample_frame(la, model):
    rows, tables = [], []
    for fold in la["folds"]:
        c = vcache.load(la, model, fold)
        T = LG.value_table(c, V, VF, V_EMPTY)
        vd = LG.as_vdict(T, V)
        terms = sh.shapley_interaction_terms(V, vd)
        P = sh.shapley_values(V, vd)
        yhat = LG.yhat_full(c, V)
        d = {"fold": fold, "sample_idx": c["sample_idx"], "lat": c["lat"], "lon": c["lon"],
             "label": c["y_true"], "prediction": yhat, "correct": (yhat == c["y_true"]).astype(np.int64)}
        for v in V:
            d[f"phi_{LAB[v]}"] = P[v]
        for pair in G.PAIRS:
            t = terms.get(pair) or terms[pair[::-1]]
            I = sum(t.values())
            d[col(pair)] = I
            d[col(pair, "I_ctx")] = I - t[0]
        rows.append(pd.DataFrame(d))
        tables.append(T)
    return pd.concat(rows, ignore_index=True), tables


def check(df, tables, rng) -> dict:
    T = np.concatenate(tables, axis=1)
    idx = rng.choice(T.shape[1], size=min(N_CHECK, T.shape[1]), replace=False)
    err_I, err_eff = 0.0, 0.0
    for k in idx:
        g = {vcache.members(m, V): float(T[m, k]) for m in range(T.shape[0])}
        d = G.harsanyi(g)
        for pair in G.PAIRS:
            via = sum(x / (len(S) - 1) for S, x in d.items() if set(pair) <= S)
            err_I = max(abs(df[col(pair)].iat[k] - via), err_I)
        eff = sum(df[f"phi_{LAB[v]}"].iat[k] for v in V) - (g[frozenset(V)] - V_EMPTY)
        err_eff = max(abs(eff), err_eff)
    if max(err_I, err_eff) > TOL:
        raise AssertionError(f"local identities violated: I via Harsanyi {err_I:.1e}, efficiency {err_eff:.1e}")
    return {"n_samples": int(len(idx)), "I_via_harsanyi": err_I, "efficiency": err_eff}


def summarise(frames: dict) -> pd.DataFrame:
    rows = []
    for model, df in frames.items():
        for name, cond in SUBSETS:
            sub = df if cond is None else df[df[cond[0]] == cond[1]]
            for pair in G.PAIRS:
                I, Cx = sub[col(pair)], sub[col(pair, "I_ctx")]
                rows.append({"model": model, "subset": name, "n": len(sub), "pair": G.pair_label(pair),
                             "I_mean": I.mean(), "I_median": I.median(), "I_neg_share": (I < 0).mean(),
                             "I_ctx_mean": Cx.mean(), "I_ctx_neg_share": (Cx < 0).mean()})
    return pd.DataFrame(rows)


def cells(df, pair=S2W) -> pd.DataFrame:
    li = np.floor((df.lon.values + 180) / RES).astype(int)
    la = np.floor((df.lat.values + 90) / RES).astype(int)
    g = df.assign(_i=li, _j=la).groupby(["_i", "_j"])
    out = g.agg(n=("lat", "size"), I_mean=(col(pair), "mean"),
                **{f"phi_{LAB[v]}_mean": (f"phi_{LAB[v]}", "mean") for v in pair}).reset_index()
    out["lon_c"] = -180 + (out._i + 0.5) * RES
    out["lat_c"] = -90 + (out._j + 0.5) * RES
    out = out[out.n >= MIN_COUNT].drop(columns=["_i", "_j"])
    return out[["lat_c", "lon_c", "n", "I_mean"] + [c for c in out if c.startswith("phi_")]]


def _maps(panels, nrows, ncols, figsize, out, cbar_label):
    from shapfusion.analysis import plotstyle as ps
    import matplotlib.pyplot as plt
    grids = [grid_mean(df.lon.values, df.lat.values, df[c].values, RES, MIN_COUNT) for _, df, c in panels]
    vmax = float(np.nanpercentile(np.abs(np.concatenate([g[0][np.isfinite(g[0])].ravel() for g in grids])), VMAX_Q))
    bg = land()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    pc = None
    for ax, (label, df, c), (mean, _, lc, ltc) in zip(axes.flatten(), panels, grids):
        setup(ax, bg, LAT_LIM)
        pc = ax.pcolormesh(lc, ltc, mean.T, cmap="RdBu", vmin=-vmax, vmax=vmax, shading="auto", zorder=2,
                           rasterized=True)
        ax.set_title(f"{label}      mean {df[c].mean():+.3f}", fontsize=9.5, color=ps.INK, loc="left", pad=4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.11, wspace=0.04, hspace=0.16)
    cb = fig.colorbar(pc, cax=fig.add_axes([0.30, 0.065, 0.40, 0.02]), orientation="horizontal")
    cb.set_label(cbar_label, fontsize=8.5, color=ps.INK)
    cb.ax.tick_params(labelsize=7.5, colors=ps.INK_MUTED, length=3)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(LAND_EDGE)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return vmax


CBAR = (r"mean local interaction $I_{{{}}}$ per {:g}° cell, red substitution ($I<0$), "
        r"blue complementarity ($I>0$)")


def figures(cfg, frames: dict) -> dict:
    figs = C.out(cfg, "figures")
    models = list(frames)
    lim = {"s2_weather": _maps([(m, frames[m], col(S2W)) for m in models], len(models), 1, (8.6, 3.5 * len(models)),
                               figs / "interactions_local_s2_weather.pdf", CBAR.format(r"\mathrm{S2,weather}", RES))}
    lim["pairs"] = _maps([(G.pair_label(p).replace("-", " × "), frames[models[0]], col(p)) for p in G.PAIRS], 3, 2,
                         (12.4, 7.4), figs / "interactions_local_pairs.pdf", CBAR.format("ij", RES))
    return lim


def _pct(x):
    return f"{100 * x:.1f}"


def markdown(cfg, frames, summary, cell_tabs, checks, vmax) -> str:
    models = list(frames)
    first = models[0]
    L = ["# Local interactions", "",
         f"Local game of the local Shapley maps: v_x(S) = P(ŷ_full(x) | x_S), v_x(∅) = **{V_EMPTY:g}** (uniform "
         "prior); every test sample is explained by the model of the fold it is a test sample of; EmbraceNet = "
         "mean of the class probabilities over the R stochastic draws. Order-2 interaction index, same formula "
         "as the global one (weights 1/3, 1/6, 1/6, 1/3): I = I_vide + I_ctx, "
         "I_vide = (1/3)[v({i,j}) − v({i}) − v({j}) + v(∅)]. **I < 0 = substitution, I > 0 = complementarity.** "
         f"{sum(len(f) for f in frames.values()) // len(frames):,} test samples ({len(cfg['folds'])} folds).", "",
         "## Checks", "",
         f"Identities on {N_CHECK} random samples per model: I_ij(x) = Σ_{{T⊇{{i,j}}}} d_x(T)/(|T|−1) "
         "(Harsanyi dividends of the sample's scalar game); efficiency Σ_i φ_i(x) = v_x(N) − 0.5.", "",
         f"| model | max \\|err\\| I via Harsanyi | max \\|err\\| efficiency | status (tolerance {TOL:g}) |",
         "|---|---|---|---|"]
    for m, c in checks.items():
        ok = max(c["I_via_harsanyi"], c["efficiency"]) <= TOL
        L.append(f"| {m} | {c['I_via_harsanyi']:.1e} | {c['efficiency']:.1e} | {'OK' if ok else 'FAILED'} |")
    L += ["", "### Mean local φ per modality (mean over the samples)", "",
          "These are the means printed on the local Shapley maps. By efficiency, Σφ = mean of v_x(N) − 0.5.", "",
          "| modality | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for v in V:
        L.append(f"| {LAB[v]} | " + " | ".join(f"{frames[m][f'phi_{LAB[v]}'].mean():+.4f}" for m in models) + " |")
    L.append("| **Σφ** | " + " | ".join(f"{sum(frames[m][f'phi_{LAB[v]}'].mean() for v in V):+.4f}"
                                        for m in models) + " |")
    L += ["", f"Colour limits of the maps (98th percentile of \\|cell mean\\|, {RES:g}° cells ≥ {MIN_COUNT} samples): "
          f"S2–weather ±{vmax['s2_weather']:.3f} (shared by the models), 6 pairs of {first} ±{vmax['pairs']:.3f}.", ""]
    for m in models:
        L += [f"## {m}", "", "### a. Local interactions per pair", ""]
        for name, _ in SUBSETS:
            d = summary[(summary.model == m) & (summary.subset == name)]
            L += [f"**{name} samples** (n = {d.n.iloc[0]:,})", "",
                  "| pair | mean I | median I | share I < 0 (%) | mean I_ctx | share I_ctx < 0 (%) |",
                  "|---|---|---|---|---|---|"]
            for r in d.itertuples():
                L.append(f"| {r.pair} | {r.I_mean:+.4f} | {r.I_median:+.4f} | {_pct(r.I_neg_share)} "
                         f"| {r.I_ctx_mean:+.4f} | {_pct(r.I_ctx_neg_share)} |")
            L.append("")
        c = cell_tabs[m]
        L += ["### b. Extreme cells of I(S2, weather)", "",
              f"{len(c):,} cells of {RES:g}° × {RES:g}° with ≥ {MIN_COUNT} samples; lat/lon = cell centre; "
              "φ = local means in the cell.", ""]
        for name, part in (("10 most negative cells", c.nsmallest(10, "I_mean")),
                           ("10 most positive cells", c.nlargest(10, "I_mean"))):
            L += [f"**{name}**", "", "| lat | lon | n | mean I | φ_S2 | φ_weather |", "|---|---|---|---|---|---|"]
            for r in part.itertuples():
                L.append(f"| {r.lat_c:+.0f} | {r.lon_c:+.0f} | {r.n} | {r.I_mean:+.4f} "
                         f"| {r.phi_S2_mean:+.4f} | {r.phi_weather_mean:+.4f} |")
            L.append("")
    return "\n".join(L)


def run(cfg):
    la = la_config.build(cfg)
    rng = np.random.default_rng(SEED)
    frames, checks = {}, {}
    for model in G.models(cfg):
        df, tables = sample_frame(la, model)
        checks[model] = check(df, tables, rng)
        frames[model] = df
    summary = summarise(frames)
    cell_tabs = {m: cells(frames[m]) for m in frames}
    data = C.out(cfg, "data")
    keep = ["fold", "sample_idx", "lat", "lon", "label", "prediction"] + [f"phi_{LAB[v]}" for v in V] + \
           [col(p) for p in G.PAIRS] + [col(p, "I_ctx") for p in G.PAIRS]
    for m, df in frames.items():
        df[keep].to_csv(data / f"interactions_local_{m}.csv.gz", index=False)
    summary.to_csv(data / "interactions_local_summary.csv", index=False)
    pd.concat([c.assign(model=m) for m, c in cell_tabs.items()]).to_csv(
        data / "interactions_local_cells_s2_weather.csv", index=False)
    vmax = figures(cfg, frames)
    (data / "interactions_local_checks.json").write_text(json.dumps(
        {"value_function": VF, "v_empty": V_EMPTY, "grid_deg": RES, "min_count": MIN_COUNT,
         "colour_limit": vmax, "identities": checks}, indent=2))
    (C.out(cfg, "numbers") / "interactions_local.md").write_text(markdown(cfg, frames, summary, cell_tabs, checks, vmax))
