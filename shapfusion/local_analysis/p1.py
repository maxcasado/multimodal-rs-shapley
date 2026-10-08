"""
P1 — robustness of the maps (tasks 6-11). Every figure is drawn from grid.cell_stats of
the prob_pred game (the game of the paper's local maps) unless stated, rows = models, one
colour scale per figure, and its table is written next to it.
"""
import numpy as np
import pandas as pd

from . import data, games, grid, maps
from .config import section


def _models(cfg):
    return list(cfg["models"])


def _row_labels(cfg, notes):
    return [data.model_label(cfg, m, notes.get(m)) for m in _models(cfg)]


def quantised_notes(cfg):
    """Row-label note for quantised models (P0.5): share of samples on the 16 most
    frequent values of p(ŷ | x)."""
    from . import p0
    notes = {}
    for m, s in cfg["models"].items():
        if s.get("quantised"):
            q = p0.quantisation(cfg, m)
            notes[m] = "(quantised*)"
            cfg["_quant_note"] = (f"*{s['label']}: 16 values of p(ŷ|x) carry "
                                  f"{100 * q.share_top16_values.mean():.1f}% of test samples")
    return notes


def _clip_rows(res, fig_name, models, cols, value_function):
    rows = []
    for r, m in enumerate(models):
        for c, q in enumerate(cols):
            g = next(g for g in res["groups"] if c in g["cols"])
            rows.append({"figure": fig_name, "model": m, "quantity": q,
                         "value_function": value_function, "vmax": g["vmax"],
                         "vmin": g["vmin"], "pct_clipped": res["clipped"][r][c]})
    return rows


def signed_map_figure(cfg, cols, stat, out, *, title, cbar_label, vf="prob_pred",
                      stipple=False, mask_inconsistent=False, signed=True, vmax=None,
                      notes=None, extra_legend="", table_stats=None):
    """rows = models, cols = quantities, cell statistic ``stat`` of each quantity."""
    models = _models(cfg)
    panels, tidy = [], []
    for m in models:
        row = []
        for c in cols:
            st = data.stats(cfg, m, c, vf)
            row.append(maps.panel(st, stat, cfg,
                                  mask=(~st.sign_consistent.values) if mask_inconsistent else None,
                                  stipple=(~st.sign_consistent.values) if stipple else None))
            tidy.append(grid.tidy(st[st.ok], m, c, table_stats or (stat,)).assign(
                value_function=vf))
        panels.append(row)
    vlabel = games.vf_label(cfg, vf)
    if vmax is None:
        vmax = maps.colour_limit(panels, cfg["colour"]["vmax_q"], signed)
    res = maps.figure(panels, out, cfg, row_labels=_row_labels(cfg, notes or {}),
                      col_titles=[data.qlabel(c) for c in cols], title=title,
                      cbar_label=cbar_label, signed=signed, vmax=vmax,
                      legend=maps.legend_text(cfg, vlabel, vmax, extra_legend))
    data.write_table(pd.concat(tidy, ignore_index=True), out.with_suffix(".csv.gz"))
    return res


# ── task 6: counts ────────────────────────────────────────────────────────────

def task06_counts(cfg):
    out = section(cfg, "p1_robustness")
    m0 = _models(cfg)[0]
    df = data.frame(cfg, m0)
    st = data.stats(cfg, m0, "phi_S2")           # any quantity: n / n_f<k> are the same
    g = cfg["grid"]
    folds = cfg["folds"]
    st = st.assign(n_folds_present=(st[[f"n_f{f}" for f in folds]] > 0).sum(1))
    n_panel = maps.panel(st, "n", cfg)
    f_panel = maps.panel(st, "n_folds_present", cfg)
    legend = (f"{g['res']:g}° cells, n_min = {g['n_min']} (hatched: 0 < n < n_min); "
              f"same test samples for every model (5 folds partition {len(df):,} samples)")
    res = maps.figure([[n_panel, f_panel]], out / "fig_cell_counts.png", cfg,
                      row_labels=["all models"], col_titles=["samples per cell (n)",
                                                             "folds present in the cell"],
                      title="Samples per grid cell", cbar_label="", legend=legend,
                      groups=[{"cols": [0], "signed": False, "log": True,
                               "vmax": float(st.n.max()), "vmin": float(g["n_min"]),
                               "cbar_label": "n (log scale)"},
                              {"cols": [1], "signed": False, "vmax": len(folds), "vmin": 0,
                               "cbar_label": "folds with >= 1 sample"}])
    base = st.reset_index()[["cell_id", "lon", "lat", "n"]]
    tab = pd.concat([base.assign(fold="pooled", value=st.n.values)] +
                    [base.assign(n=st[f"n_f{f}"].values, fold=f, value=st[f"n_f{f}"].values)
                     for f in folds], ignore_index=True)
    tab.insert(4, "model", "all")
    tab.insert(5, "quantity", "n_samples")
    data.write_table(tab, out / "fig_cell_counts.csv.gz")
    ok = st.n >= g["n_min"]
    return {"n_cells": int(len(st)), "n_cells_ge_nmin": int(ok.sum()),
            "share_samples_kept": float(st.n[ok].sum() / st.n.sum()),
            "median_n_kept": float(st.n[ok].median()),
            "cells_ge_nmin_all_folds": int((ok & (st.n_folds_present == len(folds))).sum()),
            "cells_ge_nmin_ge_sign_k_folds": int((ok & (st.n_folds_present >= g["sign_k"])).sum()),
            "clipped": res["clipped"]}


# ── tasks 7, 8, 9, 10 ─────────────────────────────────────────────────────────

def task07_10_maps(cfg):
    out = section(cfg, "p1_robustness")
    phi, sii = data.quantities(cfg)
    notes = quantised_notes(cfg)
    vlabel = games.vf_label(cfg, "prob_pred")
    clip, key = [], {}

    # E[φ] and E[I] (pooled cell mean), plain and with the sign-consistency overlay (task 7)
    for cols, name, lab in ((phi, "phi", r"mean local $\phi$ (blue: supports ŷ_full, red: against)"),
                            (sii, "sii", r"mean local $I_{ij}$ (blue: complementary, red: redundant)")):
        for overlay in (False, True):
            fn = f"fig_{name}_mean{'_signmask' if overlay else ''}.png"
            res = signed_map_figure(
                cfg, cols, "mean", out / fn, stipple=overlay, notes=notes,
                title=f"Local {'Shapley values' if name == 'phi' else 'Shapley interactions'}, "
                      "mean per grid cell" + (" — dots: sign not consistent across folds"
                                              if overlay else ""),
                cbar_label=lab,
                extra_legend=(f"dots: fold means agree in sign in < {cfg['grid']['sign_k']} "
                              f"of {len(cfg['folds'])} folds" if overlay else ""),
                table_stats=("mean", "fold_mean", "fold_std", "ratio"))
            clip += _clip_rows(res, fn, _models(cfg), cols, "prob_pred")
        # mean / std over folds (task 7)
        fn = f"fig_{name}_fold_ratio.png"
        res = signed_map_figure(
            cfg, cols, "ratio", out / fn, stipple=True, notes=notes,
            title=f"Fold stability of the local {'φ' if name == 'phi' else 'interactions'}: "
                  "mean / std of the 5 fold means per cell",
            cbar_label="mean over folds / std over folds",
            extra_legend=(f"dots: sign consistent in < {cfg['grid']['sign_k']} folds; folds "
                          "partition the samples, so the spread mixes model and sampling noise"),
            table_stats=("fold_mean", "fold_std", "ratio"))
        clip += _clip_rows(res, fn, _models(cfg), cols, "prob_pred")

    # E|φ| beside E[φ], one figure per sensor, scales shared across sensors (task 8)
    vmax_s = maps.colour_limit([[maps.panel(data.stats(cfg, m, c), "mean", cfg)
                                 for c in phi] for m in _models(cfg)],
                               cfg["colour"]["vmax_q"], True)
    vmax_a = maps.colour_limit([[maps.panel(data.stats(cfg, m, c), "abs_mean", cfg)
                                 for c in phi] for m in _models(cfg)],
                               cfg["colour"]["vmax_q"], False)
    for c in phi:
        panels, tidy = [], []
        for m in _models(cfg):
            st = data.stats(cfg, m, c)
            panels.append([maps.panel(st, "mean", cfg), maps.panel(st, "abs_mean", cfg)])
            tidy.append(grid.tidy(st[st.ok], m, c, ("mean", "abs_mean")))
        fn = f"fig_phi_mean_vs_abs_{data.qlabel(c)}.png"
        res = maps.figure(
            panels, out / fn, cfg, row_labels=_row_labels(cfg, notes),
            col_titles=[f"{data.qlabel(c)}: E[φ]", f"{data.qlabel(c)}: E|φ|"],
            title=f"{data.qlabel(c)} — signed mean vs mean magnitude of the local φ per cell",
            cbar_label="", legend=maps.legend_text(
                cfg, vlabel, vmax_s, f"E|φ| scale 0..{vmax_a:.3g} (p{cfg['colour']['vmax_q']:g}); "
                                     "both scales shared by the four sensor figures"),
            groups=[{"cols": [0], "signed": True, "vmax": vmax_s, "cbar_label": "E[φ]"},
                    {"cols": [1], "signed": False, "vmax": vmax_a, "cbar_label": "E|φ|"}])
        data.write_table(pd.concat(tidy, ignore_index=True), (out / fn).with_suffix(".csv.gz"))
        clip += _clip_rows(res, fn, _models(cfg), ["mean", "abs_mean"], "prob_pred")

    # normalised shares (task 9): E_cell[φ_i] / E_cell[Σ_j φ_j] = E[φ_i] / E[v(N) - v(∅)]
    for m in _models(cfg):
        df = data.frame(cfg, m)
        tot = df[phi].sum(1)
        for c in phi:
            key.setdefault("global_share", {}).setdefault(m, {})[c] = float(df[c].mean() / tot.mean())
    panels, tidy = [], []
    for m in _models(cfg):
        df = data.frame(cfg, m)
        den = df.assign(_tot=df[phi].sum(1))
        dst = grid.cell_stats(den, "_tot", cfg)
        row = []
        for c in phi:
            st = data.stats(cfg, m, c).copy()
            st["share"] = st["mean"] / dst["mean"]
            for f in cfg["folds"]:
                st[f"f{f}"] = st[f"f{f}"] / dst[f"f{f}"]
            M = st[[f"f{f}" for f in cfg["folds"]]].values
            st["fold_mean"] = np.nanmean(M, 1)
            st["fold_std"] = np.nanstd(M, 1, ddof=1)
            row.append(maps.panel(st, "share", cfg))
            tidy.append(grid.tidy(st[st.ok], m, c, ("share", "fold_mean", "fold_std")))
        panels.append(row)
    fn = "fig_phi_share.png"
    vmax = maps.colour_limit(panels, cfg["colour"]["vmax_q"], True)
    res = maps.figure(panels, out / fn, cfg, row_labels=_row_labels(cfg, notes),
                      col_titles=[data.qlabel(c) for c in phi],
                      title="Normalised local φ per cell: E[φ_i] / E[Σ_j φ_j] (= E[φ_i] / E[v(N) − v(∅)])",
                      cbar_label="share of the cell's total attribution", vmax=vmax,
                      legend=maps.legend_text(cfg, vlabel, vmax,
                                              "ratio of cell means (per-sample ratios blow up "
                                              "where v(N) ≈ 0.5)"))
    data.write_table(pd.concat(tidy, ignore_index=True), (out / fn).with_suffix(".csv.gz"))
    clip += _clip_rows(res, fn, _models(cfg), phi, "prob_pred")

    # task 10: clipping report
    clip = pd.DataFrame(clip)
    data.write_table(clip, out / "clipping.csv")
    key["clipped_max_pct"] = float(clip.pct_clipped.max())
    key["clipped_median_pct"] = float(clip.pct_clipped.median())

    # headline numbers for the summary: share of cells with a consistent sign, median ratio
    cons = []
    for m in _models(cfg):
        for c in phi + sii:
            st = data.stats(cfg, m, c)
            ok = st.ok & (st.n_folds >= cfg["grid"]["sign_k"])
            cons.append({"model": m, "quantity": c, "cells": int(ok.sum()),
                         "share_sign_consistent": float(st.sign_consistent[ok].mean()),
                         "median_abs_ratio": float(st.ratio[ok].abs().median()),
                         "pooled_mean_over_samples": float(data.frame(cfg, m)[c].mean())})
    cons = pd.DataFrame(cons)
    data.write_table(cons, out / "table_fold_consistency.csv")
    key["fold_consistency"] = cons.to_dict("records")
    return key


# ── task 11: regional aggregation ─────────────────────────────────────────────

def task11_regional(cfg):
    out = section(cfg, "p1_regional")
    phi, sii = data.quantities(cfg)
    rows = []
    for m in _models(cfg):
        df = data.frame(cfg, m)
        for by in ("region", "subdataset"):
            for c in phi + sii:
                a = grid.fold_agg(df, c, by, cfg["folds"]).reset_index().rename(columns={by: "group"})
                rows.append(a.assign(model=m, grouping=by, quantity=c, value_function="prob_pred"))
    tab = pd.concat(rows, ignore_index=True)[["model", "grouping", "group", "quantity", "n",
                                              "pooled_mean", "fold_mean", "fold_std", "n_folds",
                                              "value_function"]]
    data.write_table(tab, out / "table_regional.csv")
    # readable wide version: mean ± std over folds
    tab["cell"] = [data.fmt(a, b) for a, b in zip(tab.fold_mean, tab.fold_std)]
    for by in ("region", "subdataset"):
        w = tab[tab.grouping == by].pivot_table(index=["group", "n"], columns=["model", "quantity"],
                                                values="cell", aggfunc="first")
        w.to_csv(out / f"table_regional_{by}_wide.csv")
    return {"n_rows": int(len(tab))}
