"""
P4 — interpretation hypotheses (tasks 19-20), prob_pred game.

19  weather / DEM as location proxies.
    (a) share of the per-sample variance of φ explained by the grid cell and by the
        CropHarvest sub-dataset of origin: R² (and adjusted R², which corrects for the
        ~2 600 cell levels) of a one-way fixed-effect model, per fold then mean ± std.
        Every sensor is reported so weather / DEM can be read against S2 / S1.
    (b) per cell, φ_weather vs class purity |crop share − 0.5| (crop share = mean true
        label of the cell's samples): weighted Spearman, per fold then mean ± std.
20  S1 as a backup sensor: per cell, φ_S1 vs φ_S2 and φ_S1 vs I(S2, S1). No cloud-cover
    or S2-quality variable exists in cropharvest_binary.nc (variables: S2, S1, weather,
    DEM, S2VI, S1Ind, year, coords, train_mask, target), so that correlation is not done.
"""
import pandas as pd

from . import data, games, grid, maps, stats
from .config import section


def _models(cfg):
    return list(cfg["models"])


def task19_location_proxy(cfg):
    out = section(cfg, "p4_hypotheses")
    phi, _ = data.quantities(cfg)
    g = cfg["grid"]
    rows = []
    for m in _models(cfg):
        df = data.frame(cfg, m)
        for c in phi:
            for by in ("cell_id", "subdataset"):
                per = [stats.fixed_effect_r2(sub[c].values, sub[by].values)
                       for _, sub in df.groupby("fold")]
                pooled = stats.fixed_effect_r2(df[c].values, df[by].values)
                r2m, r2s = data.mean_std([p[0] for p in per])
                am, as_ = data.mean_std([p[1] for p in per])
                rows.append({"model": m, "quantity": c, "grouping": by,
                             "r2_fold_mean": r2m, "r2_fold_std": r2s,
                             "adj_r2_fold_mean": am, "adj_r2_fold_std": as_,
                             "r2_pooled": pooled[0], "adj_r2_pooled": pooled[1],
                             "n_groups_pooled": pooled[2], "n_samples": pooled[3]})
    r2 = pd.DataFrame(rows)
    data.write_table(r2, out / "table_location_r2.csv")

    # purity
    xs, ys, ws, ann, prow = [], [], [], [], []
    targets = ["phi_weather", "phi_DEM"]
    for m in _models(cfg):
        df = data.frame(cfg, m)
        pur = grid.cell_stats(df.assign(y=df.y_true.astype(float)), "y", cfg)
        for f in cfg["folds"]:
            pur[f"f{f}"] = (pur[f"f{f}"] - 0.5).abs()
        pur["purity"] = (pur["mean"] - 0.5).abs()
        rx, ry, rw, ra = [], [], [], []
        for c in targets:
            X = data.stats(cfg, m, c)
            ok = X.ok
            rho = []
            for f in cfg["folds"]:
                okf = ok & (X[f"n_f{f}"] >= g["min_per_fold"])
                rho.append(stats.wspearman(pur.loc[okf, f"f{f}"], X.loc[okf, f"f{f}"],
                                           X.loc[okf, f"n_f{f}"]))
            pooled = stats.wspearman(pur.loc[ok, "purity"], X.loc[ok, "mean"], X.loc[ok, "n"])
            mm, ss = data.mean_std(rho)
            prow.append({"model": m, "quantity": c, "x": "|crop share - 0.5|",
                         "spearman_w_fold_mean": mm, "spearman_w_fold_std": ss,
                         "spearman_w_pooled": pooled, "n_cells": int(ok.sum())})
            rx.append(pur.loc[ok, "purity"].values)
            ry.append(X.loc[ok, "mean"].values)
            rw.append(X.loc[ok, "n"].values)
            ra.append(f"ρ_w folds {data.fmt(mm, ss, 2)}\nρ_w pooled {pooled:+.2f}")
            data.write_table(pd.DataFrame({"cell_id": X.index[ok], "lon": X.lon[ok], "lat": X.lat[ok],
                                           "n": X.n[ok], "model": m, "quantity": c,
                                           "purity": pur.loc[ok, "purity"].values,
                                           "value": X.loc[ok, "mean"].values}),
                             out / "scatter" / f"purity_{m}_{c}.csv")
        xs.append(rx)
        ys.append(ry)
        ws.append(rw)
        ann.append(ra)
    pt = pd.DataFrame(prow)
    data.write_table(pt, out / "table_purity.csv")
    maps.scatter_grid(xs, ys, out / "fig_phi_vs_class_purity.png",
                      row_labels=[cfg["models"][m]["label"] for m in _models(cfg)],
                      col_titles=["φ_weather", "φ_DEM"], xlabel="cell |crop share − 0.5|",
                      ylabel="cell mean φ", weights=ws, annotate=ann,
                      title="Location-proxy check: per-cell φ vs how class-pure the cell is",
                      legend=f"{games.vf_label(cfg, 'prob_pred')}; cells with n ≥ {g['n_min']}; "
                             "point size ∝ √n; ρ_w = n-weighted Spearman")
    return {"r2": r2.round(3).to_dict("records"), "purity": pt.round(3).to_dict("records")}


def task20_s1_backup(cfg):
    out = section(cfg, "p4_hypotheses")
    g = cfg["grid"]
    pairs_ = [("phi_S2", "phi_S1"), ("I_S2xS1", "phi_S1")]
    xs, ys, ws, ann, rows = [], [], [], [], []
    for m in _models(cfg):
        Y = data.stats(cfg, m, "phi_S1")
        rx, ry, rw, ra = [], [], [], []
        for xc, yc in pairs_:
            X = data.stats(cfg, m, xc)
            ok = Y.ok
            rho = []
            for f in cfg["folds"]:
                okf = ok & (Y[f"n_f{f}"] >= g["min_per_fold"])
                rho.append(stats.wspearman(X.loc[okf, f"f{f}"], Y.loc[okf, f"f{f}"],
                                           Y.loc[okf, f"n_f{f}"]))
            pooled = stats.wspearman(X.loc[ok, "mean"], Y.loc[ok, "mean"], Y.loc[ok, "n"])
            pear = stats.wpearson(X.loc[ok, "mean"], Y.loc[ok, "mean"], Y.loc[ok, "n"])
            mm, ss = data.mean_std(rho)
            rows.append({"model": m, "x": xc, "y": yc, "spearman_w_fold_mean": mm,
                         "spearman_w_fold_std": ss, "spearman_w_pooled": pooled,
                         "pearson_w_pooled": pear, "n_cells": int(ok.sum())})
            rx.append(X.loc[ok, "mean"].values)
            ry.append(Y.loc[ok, "mean"].values)
            rw.append(Y.loc[ok, "n"].values)
            ra.append(f"ρ_w folds {data.fmt(mm, ss, 2)}\nρ_w pooled {pooled:+.2f}")
            data.write_table(pd.DataFrame({"cell_id": X.index[ok], "lon": X.lon[ok], "lat": X.lat[ok],
                                           "n": X.n[ok], "model": m, "x_quantity": xc,
                                           "x": X.loc[ok, "mean"].values,
                                           "phi_S1": Y.loc[ok, "mean"].values}),
                             out / "scatter" / f"s1_{m}_{xc}.csv")
        xs.append(rx)
        ys.append(ry)
        ws.append(rw)
        ann.append(ra)
    tab = pd.DataFrame(rows)
    data.write_table(tab, out / "table_s1_backup.csv")
    maps.scatter_grid(xs, ys, out / "fig_s1_backup.png",
                      row_labels=[cfg["models"][m]["label"] for m in _models(cfg)],
                      col_titles=["φ_S1 vs φ_S2", "φ_S1 vs I(S2, S1)"],
                      xlabel=["cell mean φ_S2", "cell mean I(S2, S1)"], ylabel="cell mean φ_S1",
                      weights=ws, annotate=ann,
                      title="S1 as a backup sensor: per-cell φ_S1 against S2's φ and the S2 × S1 interaction",
                      legend=f"{games.vf_label(cfg, 'prob_pred')}; cells with n ≥ {g['n_min']}; "
                             "no cloud / S2-quality variable in the dataset")
    return {"table": tab.round(3).to_dict("records")}
