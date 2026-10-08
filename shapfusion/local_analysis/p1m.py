"""
P1 — comparison between models (tasks 12-14), on the prob_pred game.

12  cell-by-cell correlation of every pair of models, for each sensor φ and each pair I:
    weighted Pearson and Spearman (weights = samples of the cell in the fold), computed
    on the fold-k cell means (cells with n >= n_min overall and >= min_per_fold in fold k),
    then mean ± std over folds.
13  difference maps A − B (cfg comparisons): pooled cell mean of A minus that of B; a
    cell is masked unless the fold-k differences agree in sign in >= sign_k folds.
    Same folds / same samples for every model, so the fold-k difference is paired.
14  Moran's I (stats.moran, queen contiguity on the grid,
    cells >= n_min) of every map, one summary table.
"""
import itertools

import numpy as np
import pandas as pd

from . import data, games, grid, maps, stats
from .config import section
from .p1 import _clip_rows


def _models(cfg):
    return list(cfg["models"])


def task12_correlations(cfg):
    out = section(cfg, "p1_models")
    phi, sii = data.quantities(cfg)
    g = cfg["grid"]
    rows = []
    for c in phi + sii:
        st = {m: data.stats(cfg, m, c) for m in _models(cfg)}
        for a, b in itertools.combinations(_models(cfg), 2):
            for f in cfg["folds"]:
                A, B = st[a], st[b]
                ok = A.ok & (A[f"n_f{f}"] >= g["min_per_fold"])
                x, y, w = A.loc[ok, f"f{f}"], B.loc[ok, f"f{f}"], A.loc[ok, f"n_f{f}"]
                rows.append({"quantity": c, "model_a": a, "model_b": b, "fold": f,
                             "n_cells": int(ok.sum()),
                             "pearson_w": stats.wpearson(x, y, w),
                             "spearman_w": stats.wspearman(x, y, w)})
            ok = st[a].ok
            rows.append({"quantity": c, "model_a": a, "model_b": b, "fold": "pooled",
                         "n_cells": int(ok.sum()),
                         "pearson_w": stats.wpearson(st[a].loc[ok, "mean"], st[b].loc[ok, "mean"],
                                                     st[a].loc[ok, "n"]),
                         "spearman_w": stats.wspearman(st[a].loc[ok, "mean"], st[b].loc[ok, "mean"],
                                                       st[a].loc[ok, "n"])})
    per = pd.DataFrame(rows)
    data.write_table(per, out / "table_model_correlations_per_fold.csv")
    pf = per[per.fold != "pooled"]
    summ = pf.groupby(["quantity", "model_a", "model_b"], sort=False).agg(
        pearson_mean=("pearson_w", "mean"), pearson_std=("pearson_w", "std"),
        spearman_mean=("spearman_w", "mean"), spearman_std=("spearman_w", "std"),
        n_cells_mean=("n_cells", "mean")).reset_index()
    pooled = per[per.fold == "pooled"].set_index(["quantity", "model_a", "model_b"])
    summ = summ.join(pooled[["pearson_w", "spearman_w"]].rename(
        columns={"pearson_w": "pearson_pooled", "spearman_w": "spearman_pooled"}),
        on=["quantity", "model_a", "model_b"])
    data.write_table(summ, out / "table_model_correlations.csv")
    return {"table": summ.round(3).to_dict("records")}


def diff_stats(cfg, a, b, col):
    """Cell stats of the paired difference A − B (fold-k means differenced)."""
    A, B = data.stats(cfg, a, col), data.stats(cfg, b, col)
    D = A[["lon", "lat", "n", "ok"] + [f"n_f{f}" for f in cfg["folds"]]].copy()
    D["mean"] = A["mean"] - B["mean"]
    M = np.stack([(A[f"f{f}"] - B[f"f{f}"]).values for f in cfg["folds"]], 1)
    for k, f in enumerate(cfg["folds"]):
        D[f"f{f}"] = M[:, k]
    D["n_folds"] = np.isfinite(M).sum(1)
    D["fold_mean"] = np.nanmean(M, 1)
    D["fold_std"] = np.where(D.n_folds >= 2, np.nanstd(M, 1, ddof=1), np.nan)
    D["ratio"] = D.fold_mean / D.fold_std
    D["n_pos"], D["n_neg"] = (M > 0).sum(1), (M < 0).sum(1)
    D["sign_consistent"] = np.maximum(D.n_pos, D.n_neg) >= cfg["grid"]["sign_k"]
    return D


def task13_differences(cfg):
    out = section(cfg, "p1_models")
    phi, sii = data.quantities(cfg)
    labels = {m: s["label"] for m, s in cfg["models"].items()}
    vlabel = games.vf_label(cfg, "prob_pred")
    clip, key = [], []
    for cols, name in ((phi, "phi"), (sii, "sii")):
        panels, tidy = [], []
        for a, b in cfg["comparisons"]:
            row = []
            for c in cols:
                D = diff_stats(cfg, a, b, c)
                row.append(maps.panel(D, "mean", cfg, mask=~D.sign_consistent.values))
                tidy.append(grid.tidy(D[D.ok], f"{a}-{b}", c, ("mean", "fold_mean", "fold_std")))
                ok = D.ok
                key.append({"comparison": f"{labels[a]} − {labels[b]}", "quantity": c,
                            "cells": int(ok.sum()),
                            "share_consistent": float(D.sign_consistent[ok].mean()),
                            "share_consistent_pos": float((D.sign_consistent & (D.n_pos >= D.n_neg))[ok].mean()),
                            "share_consistent_neg": float((D.sign_consistent & (D.n_neg > D.n_pos))[ok].mean()),
                            "mean_diff_samples": float(data.frame(cfg, a)[c].mean() -
                                                       data.frame(cfg, b)[c].mean())})
            panels.append(row)
        vmax = maps.colour_limit(panels, cfg["colour"]["vmax_q"], True)
        fn = f"fig_diff_{name}.png"
        res = maps.figure(
            panels, out / fn, cfg,
            row_labels=[f"{labels[a]}\n− {labels[b]}" for a, b in cfg["comparisons"]],
            col_titles=[data.qlabel(c) for c in cols],
            title=f"Difference of the local {'φ' if name == 'phi' else 'interactions'} between "
                  "models, cells with a fold-consistent sign only",
            cbar_label="difference of cell means (A − B)", vmax=vmax,
            legend=maps.legend_text(cfg, vlabel, vmax,
                                    f"grey: difference changes sign across folds (same sign "
                                    f"in < {cfg['grid']['sign_k']} of {len(cfg['folds'])})"))
        data.write_table(pd.concat(tidy, ignore_index=True), (out / fn).with_suffix(".csv.gz"))
        clip += _clip_rows(res, fn, [f"{a}-{b}" for a, b in cfg["comparisons"]], cols, "prob_pred")
    data.write_table(pd.DataFrame(clip), out / "clipping.csv")
    key = pd.DataFrame(key)
    data.write_table(key, out / "table_differences.csv")
    return {"table": key.round(3).to_dict("records")}


def moran_rows(cfg, st, stat, model, quantity, map_name, value_function="prob_pred"):
    v = st[stat].where(st.ok)
    gr = grid.to_grid(st.index.values, v.values, cfg["grid"]["res"])
    r = stats.moran(gr, cfg["moran"]["permutations"], cfg["moran"]["seed"])
    return {"model": model, "quantity": quantity, "map": map_name, "stat": stat,
            "value_function": value_function, **r}


def task14_moran(cfg):
    out = section(cfg, "p1_models")
    phi, sii = data.quantities(cfg)
    rows = []
    for m in _models(cfg):
        for c in phi + sii:
            st = data.stats(cfg, m, c)
            for stat in ("mean", "abs_mean", "ratio"):
                rows.append(moran_rows(cfg, st, stat, m, c, f"{'phi' if c in phi else 'sii'}_{stat}"))
    for a, b in cfg["comparisons"]:
        for c in phi + sii:
            D = diff_stats(cfg, a, b, c)
            rows.append(moran_rows(cfg, D, "mean", f"{a}-{b}", c, "difference"))
    mor = pd.DataFrame(rows)
    data.write_table(mor, out / "table_moran.csv")

    # summary table: one row per model × quantity (mean over folds ± std, Moran, clip)
    summ = []
    for m in _models(cfg):
        df = data.frame(cfg, m)
        for c in phi + sii:
            per = df.groupby("fold")[c].mean()
            st = data.stats(cfg, m, c)
            mm = mor[(mor.model == m) & (mor.quantity == c) & (mor.stat == "mean")].iloc[0]
            summ.append({"model": m, "quantity": c, "fold_mean": per.mean(),
                         "fold_std": per.std(ddof=1), "n_cells": int(st.ok.sum()),
                         "share_cells_sign_consistent": float(st.sign_consistent[st.ok].mean()),
                         "moran_I": mm.I, "moran_p": mm.p_value})
    summ = pd.DataFrame(summ)
    for path in (section(cfg, "p1_robustness") / "clipping.csv", out / "clipping.csv"):
        if path.exists():
            cl = pd.read_csv(path)
            cl = cl[cl.figure.isin(["fig_phi_mean.png", "fig_sii_mean.png"])]
            summ = summ.merge(cl[["model", "quantity", "pct_clipped"]].rename(
                columns={"pct_clipped": "pct_clipped_mean_map"}), on=["model", "quantity"],
                how="left") if "pct_clipped_mean_map" not in summ else summ
    data.write_table(summ, out / "table_map_summary.csv")
    return {"moran_mean_maps": mor[mor.stat == "mean"][["model", "quantity", "I", "p_value"]]
            .round(3).to_dict("records")}
