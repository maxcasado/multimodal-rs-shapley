"""
P3 — link with missing-sensor robustness (tasks 17-18), no new inference.

17  In binary, the prediction under coalition S is ŷ_full if v(S) = P(ŷ_full | x_S) >= 0.5,
    the other class otherwise, so correct_S(x) = [ŷ_S(x) = y(x)] follows from the
    prob_pred table. Accuracy per coalition (global, per fold) and per cell; loss of
    accuracy per cell for each missing sensor k: acc(N) − acc(N \\ {k}).
    EmbraceNet: v(S) is the draw-averaged probability, so ŷ_S is the decision of the
    averaged model, not of one stochastic pass (the global coalition tables average the
    METRIC over draws; both are reported side by side in the table).
18  Per cell, loss "S2 missing" vs I(S2, weather), I(S2, S1) and φ_S2: scatter + weighted
    Spearman per model (per fold, mean ± std). Hypothesis: stronger redundancy (more
    negative I) -> smaller loss, i.e. Spearman(I, loss) > 0.
"""
import numpy as np
import pandas as pd

from . import data, games, grid, maps, stats, vcache
from .config import section
from .p1 import _clip_rows, _row_labels, quantised_notes


def _models(cfg):
    return list(cfg["models"])


def correct_table(cfg, model):
    """(fold, sample_idx, cell_id, correct_<mask> for the 15 non-empty coalitions)."""
    views = cfg["views"]
    fr = data.frame(cfg, model)
    parts = []
    for fold in cfg["folds"]:
        c = vcache.load(cfg, model, fold)
        T = games.value_table(c, views, "prob_pred", cfg["v_empty"])
        yhat = games.yhat_full(c, views)
        y = c["y_true"]
        d = {"fold": fold, "sample_idx": c["sample_idx"]}
        for m in range(1, 16):
            ys = np.where(T[m] >= 0.5, yhat, 1 - yhat)
            d[f"c{m}"] = (ys == y).astype(float)
        parts.append(pd.DataFrame(d))
    out = pd.concat(parts, ignore_index=True)
    return fr[["fold", "sample_idx", "cell_id", "lon", "lat"]].merge(
        out, on=["fold", "sample_idx"], validate="one_to_one")


def coalition_name(m, views):
    return "+".join(games.short(v) for k, v in enumerate(views) if m >> k & 1)


def _global_coalition_metric(cfg, model):
    """Fold-level accuracy of the global v-dict per coalition (mean over draws for
    EmbraceNet), for side-by-side comparison; None where not stored."""
    import pickle
    views = cfg["views"]
    out = {}
    for fold in cfg["folds"]:
        vd = pickle.loads(vcache.vdict_file(cfg, model, fold).read_bytes())
        y = np.asarray(vd["y_true"]).ravel()
        for S, rec in vd["coalitions"].items():
            if not S:
                continue
            m = vcache.mask_of(S, views)
            if "pred" in rec:
                acc = float((np.asarray(rec["pred"]).argmax(1) == y).mean())
            else:
                acc = float(rec.get("metrics", {}).get("accuracy", np.nan))
            out[(fold, m)] = acc
    return out


def task17_coalition_accuracy(cfg):
    out = section(cfg, "p3_missing")
    views = cfg["views"]
    full = games.full_mask(views)
    rows, cell_rows = [], []
    notes = quantised_notes(cfg)
    panels, tidy = [], []
    missing = [(k, full & ~(1 << k)) for k in range(len(views))]
    for model in _models(cfg):
        ct = correct_table(cfg, model)
        glob = _global_coalition_metric(cfg, model)
        for fold, sub in ct.groupby("fold"):
            for m in range(1, 16):
                rows.append({"model": model, "fold": fold, "coalition": coalition_name(m, views),
                             "size": bin(m).count("1"),
                             "accuracy_from_v": float(sub[f"c{m}"].mean()),
                             "accuracy_global_vdict": glob.get((fold, m), np.nan)})
        row = []
        for k, m in missing:
            ct[f"loss_{k}"] = ct[f"c{full}"] - ct[f"c{m}"]
            st = grid.cell_stats(ct.assign(fold=ct.fold), f"loss_{k}", cfg)
            row.append(maps.panel(st, "mean", cfg))
            q = f"acc_loss_without_{games.short(views[k])}"
            tidy.append(grid.tidy(st[st.ok], model, q, ("mean", "fold_mean", "fold_std")))
            cell_rows.append(st.assign(model=model, quantity=q))
        panels.append(row)
    acc = pd.DataFrame(rows)
    data.write_table(acc, out / "table_coalition_accuracy_per_fold.csv")
    summ = acc.groupby(["model", "coalition", "size"], sort=False).agg(
        acc_mean=("accuracy_from_v", "mean"), acc_std=("accuracy_from_v", "std"),
        acc_global_mean=("accuracy_global_vdict", "mean")).reset_index()
    data.write_table(summ, out / "table_coalition_accuracy.csv")
    vmax = maps.colour_limit(panels, cfg["colour"]["vmax_q"], True)
    fn = "fig_accuracy_loss_missing.png"
    res = maps.figure(panels, out / fn, cfg, row_labels=_row_labels(cfg, notes),
                      col_titles=[f"{games.short(views[k])} missing" for k, _ in missing],
                      title="Accuracy lost per cell when one sensor is missing: "
                            "acc(all four) − acc(the other three)",
                      cbar_label="accuracy loss (red: the model gets BETTER without the sensor)",
                      vmax=vmax,
                      legend=maps.legend_text(cfg, "ŷ_S = ŷ_full if P(ŷ_full | x_S) ≥ 0.5 else "
                                                   "the other class", vmax))
    data.write_table(pd.concat(tidy, ignore_index=True), (out / fn).with_suffix(".csv.gz"))
    data.write_table(pd.DataFrame(_clip_rows(res, fn, _models(cfg),
                                             [f"loss_{games.short(v)}" for v in views],
                                             "prob_pred")), out / "clipping.csv")
    loss = []
    for model in _models(cfg):
        a = acc[acc.model == model].pivot(index="fold", columns="coalition", values="accuracy_from_v")
        allc = coalition_name(full, views)
        for k, m in missing:
            d = a[allc] - a[coalition_name(m, views)]
            loss.append({"model": model, "missing": games.short(views[k]),
                         "acc_full": a[allc].mean(), "loss_mean": d.mean(), "loss_std": d.std(ddof=1)})
    loss = pd.DataFrame(loss)
    data.write_table(loss, out / "table_accuracy_loss_global.csv")
    return {"loss": loss.round(4).to_dict("records")}


def task18_loss_vs_interactions(cfg):
    out = section(cfg, "p3_missing")
    views = cfg["views"]
    full = games.full_mask(views)
    k2 = views.index("S2_S2VI")
    m_wo = full & ~(1 << k2)
    xcols = ["I_S2xweather", "I_S2xS1", "phi_S2"]
    xs, ys, ws, ann, rows = [], [], [], [], []
    g = cfg["grid"]
    for model in _models(cfg):
        ct = correct_table(cfg, model)
        ct["loss_S2"] = ct[f"c{full}"] - ct[f"c{m_wo}"]
        L = grid.cell_stats(ct, "loss_S2", cfg)
        rx, ry, rw, ra = [], [], [], []
        for c in xcols:
            X = data.stats(cfg, model, c)
            ok = L.ok
            rho = []
            for f in cfg["folds"]:
                okf = ok & (L[f"n_f{f}"] >= g["min_per_fold"])
                rho.append(stats.wspearman(X.loc[okf, f"f{f}"], L.loc[okf, f"f{f}"],
                                           L.loc[okf, f"n_f{f}"]))
            pooled = stats.wspearman(X.loc[ok, "mean"], L.loc[ok, "mean"], L.loc[ok, "n"])
            m_, s_ = data.mean_std(rho)
            rows.append({"model": model, "x": c, "y": "acc_loss_without_S2",
                         "spearman_w_fold_mean": m_, "spearman_w_fold_std": s_,
                         "spearman_w_pooled": pooled, "n_cells": int(ok.sum())})
            rx.append(X.loc[ok, "mean"].values)
            ry.append(L.loc[ok, "mean"].values)
            rw.append(L.loc[ok, "n"].values)
            ra.append(f"ρ_w folds {data.fmt(m_, s_, 2)}\nρ_w pooled {pooled:+.2f}")
            data.write_table(pd.DataFrame({"cell_id": X.index[ok], "lon": X.lon[ok], "lat": X.lat[ok],
                                           "n": X.n[ok], "model": model, "x_quantity": c,
                                           "x": X.loc[ok, "mean"].values,
                                           "loss_S2": L.loc[ok, "mean"].values}),
                             out / "scatter" / f"{model}_{c}.csv")
        xs.append(rx)
        ys.append(ry)
        ws.append(rw)
        ann.append(ra)
    tab = pd.DataFrame(rows)
    data.write_table(tab, out / "table_loss_vs_interactions.csv")
    maps.scatter_grid(xs, ys, out / "fig_loss_vs_interactions.png",
                      row_labels=[cfg["models"][m]["label"] for m in _models(cfg)],
                      col_titles=["I(S2, weather)", "I(S2, S1)", "φ_S2"],
                      xlabel=["cell mean I(S2, weather)", "cell mean I(S2, S1)",
                              "cell mean φ_S2"],
                      ylabel="acc. loss, S2 missing",
                      title="Per-cell accuracy lost without S2 vs S2's interactions and φ "
                            "(hypothesis: more redundancy → smaller loss, ρ(I, loss) > 0)",
                      weights=ws, annotate=ann,
                      legend=f"{games.vf_label(cfg, 'prob_pred')}; cells with n ≥ {g['n_min']}; "
                             "point size ∝ √n; ρ_w = n-weighted Spearman")
    return {"table": tab.round(3).to_dict("records")}
