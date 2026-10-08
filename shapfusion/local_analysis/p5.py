"""
P5 — decomposition of the interactions (tasks 21-24).

21  I_ij = 1/3 Δ_ij(∅) + 1/6 [Δ_ij({k}) + Δ_ij({l})] + 1/3 Δ_ij({k,l}): one map per term
    (shapley.shapley_interaction_terms), prob_pred game. Δ_ij(∅) = v(ij) − v(i) − v(j) + v(∅)
    is the agreement of the two sensors ALONE.
22  DSensD+ (mean of branch logits): in log-odds the context-free term is exactly the
    agreement term −(ℓ_i + ℓ_j − 2b)/6, computed here from the branch log-odds only and
    set beside the measured log-odds I_ij, with their correlation.
23  normalised interaction per cell: E[I_ij] / (E|φ_i| + E|φ_j|).
24  complementarity: cells with E[I_ij] > 0 and a positive sign in >= sign_k folds, listed
    with their region / sub-dataset; dedicated map of the pairs involving DEM.
"""
import numpy as np
import pandas as pd

from . import data, games, grid, maps, stats
from .config import section
from .p1 import _clip_rows, _row_labels, quantised_notes

TERM_LABELS = ["1/3 Δ(∅)", "1/6 [Δ({k}) + Δ({l})]", "1/3 Δ({k,l})"]


def _models(cfg):
    return list(cfg["models"])


def task21_context(cfg):
    out = section(cfg, "p5_interactions")
    _, sii = data.quantities(cfg)
    vlabel = games.vf_label(cfg, "prob_pred")
    quantised_notes(cfg)
    figs, rows = {}, []
    for m in _models(cfg):
        panels, tidy = [], []
        df = data.frame(cfg, m)
        for s in range(3):
            row = []
            for c in sii:
                col = f"T{s}_{c[2:]}"
                st = data.stats(cfg, m, col)
                row.append(maps.panel(st, "mean", cfg))
                tidy.append(grid.tidy(st[st.ok], m, f"{c}|{TERM_LABELS[s]}",
                                      ("mean", "fold_mean", "fold_std")))
                per = df.groupby("fold")[col].mean()
                tot = df.groupby("fold")[c].mean()
                rows.append({"model": m, "pair": c, "term": TERM_LABELS[s], "context_size": s,
                             "fold_mean": per.mean(), "fold_std": per.std(ddof=1),
                             "share_of_I_fold_mean": float((per / tot).mean()),
                             "share_of_I_fold_std": float((per / tot).std(ddof=1)),
                             "neg_share_samples": float((df[col] < 0).mean())})
            panels.append(row)
        figs[m] = (panels, tidy)
    vmax = maps.colour_limit([r for p, _ in figs.values() for r in p], cfg["colour"]["vmax_q"], True)
    clip = []
    for m, (panels, tidy) in figs.items():
        fn = f"fig_sii_context_{m}.png"
        res = maps.figure(panels, out / fn, cfg, row_labels=TERM_LABELS,
                          col_titles=[data.qlabel(c) for c in sii],
                          title=f"Interaction index split by context size — {cfg['models'][m]['label']}",
                          cbar_label="mean term per cell (red: redundant, blue: complementary)",
                          vmax=vmax, legend=maps.legend_text(
                              cfg, vlabel, vmax, "one scale for the four model figures"))
        data.write_table(pd.concat(tidy, ignore_index=True), (out / fn).with_suffix(".csv.gz"))
        clip += _clip_rows(res, fn, TERM_LABELS, sii, "prob_pred")
    tab = pd.DataFrame(rows)
    data.write_table(tab, out / "table_sii_context.csv")
    data.write_table(pd.DataFrame(clip), out / "clipping_context.csv")
    return {"table": tab.round(4).to_dict("records")}


def task22_agreement(cfg):
    out = section(cfg, "p5_interactions")
    views = cfg["views"]
    _, sii = data.quantities(cfg)
    b = games._logit(float(cfg["v_empty"]))
    rows, figs = [], {}
    for m in cfg["agreement_models"]:
        L = data.frame(cfg, m, "logodds_pred")
        from . import vcache
        ags = []
        for fold in cfg["folds"]:
            c = vcache.load(cfg, m, fold)
            T = games.value_table(c, views, "logodds_pred", cfg["v_empty"])
            ags.append({p: -(T[1 << views.index(p[0])] + T[1 << views.index(p[1])] - 2 * b) / 6
                        for p in games.pairs(views)})
        panels_a, panels_i, tidy = [], [], []
        for (a_, b_), c in zip(games.pairs(views), sii):
            ag = np.concatenate([d[(a_, b_)] for d in ags])
            L[f"AG_{c[2:]}"] = ag
            # = 1/3 Δ(∅) of the log-odds game exactly when the fusion is a mean of logits;
            # on the stored float32 (TF32) outputs the gap is float noise — reported, not assumed
            gap = float(np.abs(ag - L[f"T0_{c[2:]}"].values).max())
            sa = grid.cell_stats(grid.add_cells(L, cfg), f"AG_{c[2:]}", cfg)
            si = grid.cell_stats(grid.add_cells(L, cfg), c, cfg)
            panels_a.append(maps.panel(sa, "mean", cfg))
            panels_i.append(maps.panel(si, "mean", cfg))
            tidy += [grid.tidy(sa[sa.ok], m, f"{c}|agreement_term", ("mean",)),
                     grid.tidy(si[si.ok], m, f"{c}|measured_logodds", ("mean",))]
            ok = sa.ok
            rows.append({"model": m, "pair": c,
                         "sample_pearson": float(np.corrcoef(ag, L[c].values)[0, 1]),
                         "sample_spearman": float(pd.Series(ag).corr(L[c], method="spearman")),
                         "cell_pearson_w": stats.wpearson(sa.loc[ok, "mean"], si.loc[ok, "mean"], sa.loc[ok, "n"]),
                         "cell_spearman_w": stats.wspearman(sa.loc[ok, "mean"], si.loc[ok, "mean"], sa.loc[ok, "n"]),
                         "mean_agreement": float(ag.mean()), "mean_I": float(L[c].mean()),
                         "agreement_share_of_I": float(ag.mean() / L[c].mean()),
                         "max_abs_gap_agreement_vs_measured_T0": gap})
        figs[m] = ([panels_a, panels_i], tidy)
    vmax = maps.colour_limit([r for p, _ in figs.values() for r in p], cfg["colour"]["vmax_q"], True)
    for m, (panels, tidy) in figs.items():
        fn = f"fig_agreement_term_{m}.png"
        maps.figure(panels, out / fn, cfg,
                    row_labels=["agreement term\n−(ℓ_i + ℓ_j − 2b)/6", "measured I_ij\n(log-odds)"],
                    col_titles=[data.qlabel(c) for c in sii],
                    title=f"Agreement term vs measured log-odds interaction — {cfg['models'][m]['label']}",
                    cbar_label="mean per cell, log-odds units", vmax=vmax,
                    legend=maps.legend_text(cfg, games.vf_label(cfg, "logodds_pred"), vmax,
                                            "one scale for both DSensD+ figures"))
        data.write_table(pd.concat(tidy, ignore_index=True), (out / fn).with_suffix(".csv.gz"))
    tab = pd.DataFrame(rows)
    data.write_table(tab, out / "table_agreement_term.csv")
    return {"table": tab.round(3).to_dict("records")}


def task23_normalised(cfg):
    out = section(cfg, "p5_interactions")
    views = cfg["views"]
    notes = quantised_notes(cfg)
    panels, tidy, rows = [], [], []
    for m in _models(cfg):
        row = []
        for (a, b), c in zip(games.pairs(views), data.quantities(cfg)[1]):
            I = data.stats(cfg, m, c).copy()
            A = data.stats(cfg, m, f"phi_{games.short(a)}")
            B = data.stats(cfg, m, f"phi_{games.short(b)}")
            I["norm"] = I["mean"] / (A["abs_mean"] + B["abs_mean"])
            row.append(maps.panel(I, "norm", cfg))
            tidy.append(grid.tidy(I[I.ok], m, c, ("norm",)))
            ok = I.ok
            rows.append({"model": m, "pair": c, "median_norm": float(I.norm[ok].median()),
                         "mean_norm_w": float(np.average(I.norm[ok], weights=I.n[ok])),
                         "share_cells_below_minus_0.25": float((I.norm[ok] < -0.25).mean())})
        panels.append(row)
    vmax = maps.colour_limit(panels, cfg["colour"]["vmax_q"], True)
    fn = "fig_sii_normalised.png"
    res = maps.figure(panels, out / fn, cfg, row_labels=_row_labels(cfg, notes),
                      col_titles=[data.qlabel(c) for c in data.quantities(cfg)[1]],
                      title="Normalised interaction per cell: E[I_ij] / (E|φ_i| + E|φ_j|)",
                      cbar_label="normalised interaction (red: redundant, blue: complementary)",
                      vmax=vmax, legend=maps.legend_text(cfg, games.vf_label(cfg, "prob_pred"), vmax))
    data.write_table(pd.concat(tidy, ignore_index=True), (out / fn).with_suffix(".csv.gz"))
    data.write_table(pd.DataFrame(_clip_rows(res, fn, _models(cfg), data.quantities(cfg)[1],
                                             "prob_pred")), out / "clipping_normalised.csv")
    tab = pd.DataFrame(rows)
    data.write_table(tab, out / "table_sii_normalised.csv")
    return {"table": tab.round(3).to_dict("records")}


def task24_complementarity(cfg):
    out = section(cfg, "p5_interactions")
    _, sii = data.quantities(cfg)
    k = cfg["grid"]["sign_k"]
    rows, summ = [], []
    for m in _models(cfg):
        df = data.frame(cfg, m)
        reg = df.groupby("cell_id").region.agg(lambda s: s.value_counts().index[0])
        sub = df.groupby("cell_id").subdataset.agg(lambda s: s.value_counts().index[0])
        for c in sii:
            st = data.stats(cfg, m, c)
            sel = st.ok & (st["mean"] > 0) & (st.n_pos >= k)
            t = st[sel].reset_index()[["cell_id", "lon", "lat", "n", "mean", "fold_mean",
                                       "fold_std", "n_pos", "n_folds"]]
            t.insert(4, "model", m)
            t.insert(5, "pair", c)
            t["region"] = reg.reindex(t.cell_id).values
            t["subdataset"] = sub.reindex(t.cell_id).values
            rows.append(t)
            summ.append({"model": m, "pair": c, "cells": int(st.ok.sum()), "complementary_cells": int(sel.sum()),
                         "share": float(sel.sum() / st.ok.sum()),
                         "top_regions": "; ".join(f"{r} {n}" for r, n in
                                                  t.region.value_counts().head(3).items())})
    cells = pd.concat(rows, ignore_index=True)
    data.write_table(cells, out / "table_complementary_cells.csv")
    summ = pd.DataFrame(summ)
    data.write_table(summ, out / "table_complementarity_summary.csv")

    dem = [c for c in sii if "DEM" in c]
    notes = quantised_notes(cfg)
    panels = []
    for m in _models(cfg):
        row = []
        for c in dem:
            st = data.stats(cfg, m, c)
            keep = (st["mean"] > 0) & (st.n_pos >= k)
            row.append(maps.panel(st, "mean", cfg, mask=~keep.values))
        panels.append(row)
    vmax = maps.colour_limit(panels, cfg["colour"]["vmax_q"], False)
    maps.figure(panels, out / "fig_complementarity_DEM.png", cfg, row_labels=_row_labels(cfg, notes),
                col_titles=[data.qlabel(c) for c in dem],
                title="Complementary cells of the pairs with DEM (E[I] > 0, positive in ≥ "
                      f"{k} of {len(cfg['folds'])} folds)",
                cbar_label="mean local I_ij (complementary cells only)", signed=False, vmax=vmax,
                cmap="Blues",
                legend=maps.legend_text(cfg, games.vf_label(cfg, "prob_pred"), vmax,
                                        "grey: not fold-consistently complementary"))
    data.write_table(cells[cells.pair.isin(dem)], out / "fig_complementarity_DEM.csv")
    return {"summary": summ.round(4).to_dict("records")}
