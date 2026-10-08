"""
P2 — value-function variants (tasks 15-16), from the same cached per-sample games.

15  logodds_pred: v(S) = log-odds of ŷ_full given x_S, v(∅) = b = 0. Probability is a
    squashed log-odds; where P(ŷ | x_S) saturates near 1 for several coalitions, the
    probability game's second differences shrink toward 0 or turn redundant although the
    log-odds evidence is additive. What is redundant in probability but not in log-odds
    is attributed to saturation. Compared per confidence bin of v(N).
16  prob_true: v(S) = P(y_true | x_S). For a binary task and v(∅) = 0.5 this is EXACTLY the
    prob_pred game on correctly classified samples and its mirror image (v -> 1 - v, so
    φ and I flip sign) on misclassified ones. The difference between the two maps is
    therefore carried entirely by the errors: it separates "the sensor disagrees with the
    decision" (prob_pred < 0) from "the sensor makes the model wrong" (prob_true < 0 on
    an error = the sensor pushed towards the wrong class).
"""
import numpy as np
import pandas as pd

from . import data, games, grid, maps, stats
from .config import section
from .p1 import _clip_rows, _row_labels, quantised_notes


def _models(cfg):
    return list(cfg["models"])


def _variant_maps(cfg, vf, out, stem, title_vf):
    phi, sii = data.quantities(cfg)
    notes = quantised_notes(cfg)
    vlabel = games.vf_label(cfg, vf)
    clip = []
    for cols, name in ((phi, "phi"), (sii, "sii")):
        panels, tidy = [], []
        for m in _models(cfg):
            row = []
            for c in cols:
                st = data.stats(cfg, m, c, vf)
                row.append(maps.panel(st, "mean", cfg))
                tidy.append(grid.tidy(st[st.ok], m, c, ("mean", "fold_mean", "fold_std"))
                            .assign(value_function=vf))
            panels.append(row)
        vmax = maps.colour_limit(panels, cfg["colour"]["vmax_q"], True)
        fn = f"fig_{name}_{stem}.png"
        res = maps.figure(panels, out / fn, cfg, row_labels=_row_labels(cfg, notes),
                          col_titles=[data.qlabel(c) for c in cols],
                          title=f"Local {'Shapley values' if name == 'phi' else 'interactions'}, "
                                f"mean per cell — {title_vf}",
                          cbar_label=("mean local φ" if name == "phi" else "mean local $I_{ij}$")
                          + f" ({title_vf})", vmax=vmax,
                          legend=maps.legend_text(cfg, vlabel, vmax))
        data.write_table(pd.concat(tidy, ignore_index=True), (out / fn).with_suffix(".csv.gz"))
        clip += _clip_rows(res, fn, _models(cfg), cols, vf)
    return clip


def _compare(cfg, vf_a, vf_b):
    """Per model × quantity: sample means, share negative, cell-level weighted
    Spearman between the two games (per fold, mean ± std)."""
    phi, sii = data.quantities(cfg)
    g = cfg["grid"]
    rows = []
    for m in _models(cfg):
        A, B = data.frame(cfg, m, vf_a), data.frame(cfg, m, vf_b)
        for c in phi + sii:
            sa, sb = data.stats(cfg, m, c, vf_a), data.stats(cfg, m, c, vf_b)
            rho = []
            for f in cfg["folds"]:
                ok = sa.ok & (sa[f"n_f{f}"] >= g["min_per_fold"])
                rho.append(stats.wspearman(sa.loc[ok, f"f{f}"], sb.loc[ok, f"f{f}"],
                                           sa.loc[ok, f"n_f{f}"]))
            pa = A.groupby("fold")[c].mean()
            pb = B.groupby("fold")[c].mean()
            # share of the total |attribution| of the game carried by this quantity
            tot_a = A[phi if c in phi else sii].abs().sum(1).mean()
            tot_b = B[phi if c in phi else sii].abs().sum(1).mean()
            rows.append({"model": m, "quantity": c,
                         f"mean_{vf_a}": pa.mean(), f"std_{vf_a}": pa.std(ddof=1),
                         f"mean_{vf_b}": pb.mean(), f"std_{vf_b}": pb.std(ddof=1),
                         f"abs_share_{vf_a}": float(A[c].abs().mean() / tot_a),
                         f"abs_share_{vf_b}": float(B[c].abs().mean() / tot_b),
                         f"neg_share_{vf_a}": float((A[c] < 0).mean()),
                         f"neg_share_{vf_b}": float((B[c] < 0).mean()),
                         "sample_spearman": float(pd.Series(A[c].values).corr(
                             pd.Series(B[c].values), method="spearman")),
                         "cell_spearman_mean": float(np.nanmean(rho)),
                         "cell_spearman_std": float(np.nanstd(rho, ddof=1))})
    return pd.DataFrame(rows)


def task15_logodds(cfg):
    out = section(cfg, "p2_value_function")
    clip = _variant_maps(cfg, "logodds_pred", out, "logodds", "log-odds of ŷ_full")
    comp = _compare(cfg, "prob_pred", "logodds_pred")
    data.write_table(comp, out / "table_logodds_vs_prob.csv")

    # saturation: redundancy (I < 0) share per confidence bin of v(N) = P(ŷ_full | x)
    phi, sii = data.quantities(cfg)
    edges = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0 + 1e-12])
    rows = []
    for m in _models(cfg):
        P, L = data.frame(cfg, m, "prob_pred"), data.frame(cfg, m, "logodds_pred")
        b = np.digitize(P.v_full.values, edges) - 1
        for k in range(len(edges) - 1):
            sel = b == k
            if sel.sum() == 0:
                continue
            for c in sii:
                rows.append({"model": m, "conf_lo": edges[k], "conf_hi": min(edges[k + 1], 1.0),
                             "n": int(sel.sum()), "quantity": c,
                             "neg_share_prob": float((P[c].values[sel] < 0).mean()),
                             "neg_share_logodds": float((L[c].values[sel] < 0).mean()),
                             "mean_prob": float(P[c].values[sel].mean()),
                             "mean_logodds": float(L[c].values[sel].mean()),
                             "fold_std_neg_prob": float(pd.Series(P[c].values[sel] < 0)
                                                        .groupby(P.fold.values[sel]).mean().std()),
                             "fold_std_neg_logodds": float(pd.Series(L[c].values[sel] < 0)
                                                           .groupby(L.fold.values[sel]).mean().std())})
    sat = pd.DataFrame(rows)
    data.write_table(sat, out / "fig_saturation_bins.csv")
    _saturation_figure(cfg, sat, out / "fig_saturation_bins.png")
    data.write_table(pd.DataFrame(clip), out / "clipping_logodds.csv")
    return {"comparison": comp.round(3).to_dict("records")}


def _saturation_figure(cfg, sat, path):
    import matplotlib.pyplot as plt
    from shapfusion.analysis import plotstyle
    phi, sii = data.quantities(cfg)
    models = _models(cfg)
    fig, axes = plt.subplots(len(models), len(sii), figsize=(2.6 * len(sii), 2.1 * len(models)),
                             squeeze=False, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.12, wspace=0.12, hspace=0.45)
    for r, m in enumerate(models):
        for c, q in enumerate(sii):
            ax = axes[r, c]
            s = sat[(sat.model == m) & (sat.quantity == q)]
            x = np.arange(len(s))
            ax.plot(x, s.neg_share_prob, "o-", ms=3, lw=1.2, color="#2f7ac4", label="probability")
            ax.plot(x, s.neg_share_logodds, "s-", ms=3, lw=1.2, color="#b52d20", label="log-odds")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{lo:.2f}" for lo in s.conf_lo], fontsize=6, rotation=45)
            ax.set_ylim(0, 1)
            plotstyle.style_axes(ax, labelsize=6.5)
            if r == 0:
                ax.set_title(data.qlabel(q), fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"{cfg['models'][m]['label']}\nshare with I < 0", fontsize=7.5)
            if r == len(models) - 1:
                ax.set_xlabel("P(ŷ_full | x), bin lower edge", fontsize=7.5)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Share of redundant samples (I < 0) by confidence of the full model — "
                 "probability vs log-odds value function", fontsize=10, x=0.01, ha="left")
    fig.text(0.5, 0.01, f"{games.vf_label(cfg, 'prob_pred')}  vs  "
                        f"{games.vf_label(cfg, 'logodds_pred')}; all test samples, 5 folds",
             ha="center", fontsize=7.4)
    plotstyle.save_fig(fig, path)


def task16_true_class(cfg):
    out = section(cfg, "p2_value_function")
    clip = _variant_maps(cfg, "prob_true", out, "true_class", "P(y_true | x_S)")
    comp = _compare(cfg, "prob_pred", "prob_true")
    phi, sii = data.quantities(cfg)
    # exactness of the error-only difference + where it sits
    extra, panels, tidy = [], [], []
    notes = quantised_notes(cfg)
    for m in _models(cfg):
        P, T = data.frame(cfg, m, "prob_pred"), data.frame(cfg, m, "prob_true")
        wrong = P.correct.values == 0
        sign_err = float(max(np.abs(T[c].values - np.where(wrong, -1, 1) * P[c].values).max()
                             for c in phi + sii))
        row = []
        err_rate = grid.cell_stats(P.assign(err=1 - P.correct), "err", cfg)
        for c in phi:
            # φ_true − φ_pred = −2 φ_pred on errors, 0 elsewhere
            d = P.assign(_d=T[c].values - P[c].values)
            st = grid.cell_stats(d, "_d", cfg)
            row.append(maps.panel(st, "mean", cfg))
            tidy.append(grid.tidy(st[st.ok], m, f"{c}_true_minus_pred", ("mean",)))
            ok = st.ok
            extra.append({"model": m, "quantity": c, "error_rate": float(wrong.mean()),
                          "max_abs_identity_err": sign_err,
                          "mean_pred_on_errors": float(P[c].values[wrong].mean()),
                          "share_errors_where_sensor_pushed_wrong": float((T[c].values[wrong] < 0).mean()),
                          "cell_spearman_diff_vs_error_rate": stats.wspearman(
                              st.loc[ok, "mean"], err_rate.loc[ok, "mean"], st.loc[ok, "n"])})
        panels.append(row)
    vmax = maps.colour_limit(panels, cfg["colour"]["vmax_q"], True)
    fn = "fig_phi_true_minus_pred.png"
    res = maps.figure(panels, out / fn, cfg, row_labels=_row_labels(cfg, notes),
                      col_titles=[data.qlabel(c) for c in phi],
                      title="φ under P(y_true | x_S) minus φ under P(ŷ_full | x_S), mean per cell "
                            "(non-zero only on misclassified samples)",
                      cbar_label="E[φ_true − φ_pred]  (red: the sensor supported a wrong decision)",
                      vmax=vmax, legend=maps.legend_text(cfg, "prob_true − prob_pred, v(∅) = 0.5",
                                                         vmax))
    data.write_table(pd.concat(tidy, ignore_index=True), (out / fn).with_suffix(".csv.gz"))
    clip += _clip_rows(res, fn, _models(cfg), phi, "prob_true-prob_pred")
    data.write_table(comp.merge(pd.DataFrame(extra), on=["model", "quantity"], how="left"),
                     out / "table_true_vs_pred.csv")
    data.write_table(pd.DataFrame(clip), out / "clipping_true_class.csv")
    return {"errors": pd.DataFrame(extra).round(4).to_dict("records")}
