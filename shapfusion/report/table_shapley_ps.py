"""
tables/tables_shapley_ps.txt: Shapley attribution vs Perceptual Score, one LaTeX table per
model (main experiment), plus a side-by-side normalised table and mean ± std appendix tables.

phi~ and PS~ are each quantity expressed PER FOLD as a share of its own total over the four
modalities, then averaged over the folds: both columns sum to 100 %. For phi that total is
the attainable range Delta = v(N) - v(∅) (efficiency), with the paper's v(∅) = 0.5127.
"""
import numpy as np
import pandas as pd

from shapfusion import config as C
from shapfusion.analysis import games as G

SNR_THRESHOLD = 2.0
VIEW_ORDER = ["S2_S2VI", "weather", "S1", "DEM"]
VIEW_LABEL = {"S2_S2VI": "S2", "weather": "Weather", "S1": "S1", "DEM": "DEM"}
PRETTY = {"CoM": "CoM (decision-level average)", "EmbraceNet": "EmbraceNet (feature-level)"}


def stats(cfg, model):
    """Per-fold normalisation -> mean/std/SNR per modality, plus a Sum row."""
    games = G.load_games(cfg, models_=(model,), experiments=("main",))
    full = frozenset(G.VIEW_ORDER)
    dec = G.decomposition_frame(games).rename(columns={"fold": "fold_id"})[
        ["fold_id", "modality", "phi", "phi_grand", "phi_small"]]
    ps = G.load_ps(cfg, model, "main").rename(columns={"fold": "fold_id"})[["fold_id", "modality", "PS_raw"]]
    vfull = pd.DataFrame({"fold_id": games["fold"], "v_full": [g[full] for g in games["game"]]})
    df = dec.merge(ps, on=["fold_id", "modality"]).merge(vfull, on="fold_id")
    df["v_empty"] = G.V_EMPTY_F1
    df["delta"] = df["v_full"] - df["v_empty"]
    df["phi_n"] = df["phi"] / df.groupby("fold_id")["phi"].transform("sum")
    df["ps_n"] = df["PS_raw"] / df.groupby("fold_id")["PS_raw"].transform("sum")
    tot = df.groupby("fold_id").agg(phi=("phi", "sum"), PS_raw=("PS_raw", "sum"), phi_grand=("phi_grand", "sum"),
                                    phi_small=("phi_small", "sum"), phi_n=("phi_n", "sum"), ps_n=("ps_n", "sum"),
                                    delta=("delta", "first")).reset_index()
    tot["modality"] = "Sum"
    rows = {}
    for name, g in list(df.groupby("modality")) + [("Sum", tot)]:
        r = {}
        for c in ["phi", "PS_raw", "phi_grand", "phi_small", "phi_n", "ps_n", "delta"]:
            m, s = float(g[c].mean()), float(g[c].std(ddof=1))
            r[c], r[c + "_std"], r[c + "_snr"] = m, s, (m / s if s > 0 else np.inf)
        r["ratio_grand"] = r["phi_grand"] / r["phi"] if r["phi"] else np.nan
        rows[name] = r
    return rows


def f(val, snr, dec=3, pct=False):
    """A number, with a dagger when |SNR| < 2 (not resolved)."""
    txt = f"{100 * val:.1f}\\%" if pct else f"{val:.{dec}f}"
    return txt + (r"$^{\dagger}$" if abs(snr) < SNR_THRESHOLD else "")


def table(model, rows):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Shapley attribution and Perceptual Score for %s. "
             r"$\phi$ and PS are in $f_1^{\mathrm{w}}$ points; $\tilde{\phi}$ and "
             r"$\widetilde{\mathrm{PS}}$ are the same quantities expressed per fold as a "
             r"share of their own total over the four modalities, so each column sums to "
             r"100\%% and is comparable across architectures (for $\phi$ that total is the "
             r"attainable range $\Delta = v(N)-v(\emptyset) = %.3f$, by efficiency). "
             r"$\phi = \phi^{\mathrm{grand}} + "
             r"\phi^{\mathrm{small}}$. Means over 5 folds; $^{\dagger}$ marks entries with "
             r"$|\mathrm{SNR}| < 2$ (not resolved).}" % (PRETTY.get(model, model), rows["Sum"]["delta"]),
             r"\label{tab:shapley_ps_%s}" % model.lower(), r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & $\phi$ & $\tilde{\phi}$ & PS & $\widetilde{\mathrm{PS}}$ & "
             r"$\phi^{\mathrm{grand}}$ & $\phi^{\mathrm{small}}$ & $\phi^{\mathrm{grand}}/\phi$ \\", r"\midrule"]
    for v in VIEW_ORDER + ["Sum"]:
        if v == "Sum":
            lines.append(r"\midrule")
        r = rows[v]
        ratio_snr = min(abs(r["phi_grand_snr"]), abs(r["phi_snr"]))
        lines.append(" {:<8} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
            "Sum" if v == "Sum" else VIEW_LABEL[v], f(r["phi"], r["phi_snr"]), f(r["phi_n"], r["phi_snr"], pct=True),
            f(r["PS_raw"], r["PS_raw_snr"]), f(r["ps_n"], r["PS_raw_snr"], pct=True),
            f(r["phi_grand"], r["phi_grand_snr"], dec=4), f(r["phi_small"], r["phi_small_snr"]),
            f(r["ratio_grand"], ratio_snr, pct=True)))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def extended(rows):
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & $\phi$ & $\tilde{\phi}$ & PS & $\widetilde{\mathrm{PS}}$ & $\phi^{\mathrm{grand}}/\phi$ \\", r"\midrule"]
    for v in VIEW_ORDER + ["Sum"]:
        if v == "Sum":
            lines.append(r"\midrule")
        r = rows[v]
        lines.append(" {:<8} & ${:.3f} \\pm {:.3f}$ & ${:.1f} \\pm {:.1f}$\\% & ${:.3f} \\pm {:.3f}$ & "
                     "${:.1f} \\pm {:.1f}$\\% & {:.1f}\\% \\\\".format(
                         "Sum" if v == "Sum" else VIEW_LABEL[v], r["phi"], r["phi_std"], 100 * r["phi_n"],
                         100 * r["phi_n_std"], r["PS_raw"], r["PS_raw_std"], 100 * r["ps_n"], 100 * r["ps_n_std"],
                         100 * r["ratio_grand"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def side_by_side(cfg, allrows):
    models = G.models(cfg)
    lines = [r"\begin{tabular}{l" + "rrr" * len(models) + "}", r"\toprule",
             "& " + " & ".join(r"\multicolumn{3}{c}{%s}" % m for m in models) + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(models))),
             " & " + " & ".join([r"$\tilde{\phi}$ & $\widetilde{\mathrm{PS}}$ & $\phi^{\mathrm{grand}}/\phi$"] * len(models))
             + r" \\", r"\midrule"]
    for v in VIEW_ORDER + ["Sum"]:
        if v == "Sum":
            lines.append(r"\midrule")
        cells = []
        for key in models:
            r = allrows[key][v]
            ratio_snr = min(abs(r["phi_grand_snr"]), abs(r["phi_snr"]))
            cells += [f(r["phi_n"], r["phi_snr"], pct=True), f(r["ps_n"], r["PS_raw_snr"], pct=True),
                      f(r["ratio_grand"], ratio_snr, pct=True)]
        lines.append(" {:<8} & {} \\\\".format("Sum" if v == "Sum" else VIEW_LABEL[v], " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def run(cfg):
    allrows = {m: stats(cfg, m) for m in G.models(cfg)}
    out = ["=" * 78, "Shapley / Perceptual-Score tables -- " + " and ".join(G.models(cfg)),
           "Generated by `python -m shapfusion report --only C`.", ""]
    out += [f"  {k:<11} Delta = v(N) - v(0) = {allrows[k]['Sum']['delta']:.4f}" for k in allrows]
    out += ["", "Definitions",
            "  phi          Shapley value, characteristic function v = f1_weighted,",
            "               v(0) = 0.5127, uniform classifier 2p^2/(2p+1) + 2(1-p)^2/(3-2p),",
            "               p = 0.65744 = positive share of the union of the 5 test folds.",
            "  PS           Perceptual Score (Gat et al. 2021), raw: f1(all) - f1(modality permuted),",
            f"               {cfg['perceptual_score']['n_perm']} permutations per modality and fold.",
            "  phi~, PS~    share of the OWN per-fold total, then averaged: phi~ = phi / sum_j phi,",
            "               PS~ = PS_raw / sum_j PS_raw. Both columns sum to 100 % by construction.",
            "               For phi the denominator is Delta = v(N)-v(0) (efficiency); for PS it is",
            "               just the total permutation drop, which obeys no such identity.",
            "  phi_grand    the grand-coalition term of phi, i.e. [v(N) - v(N minus m)] / |N|.",
            "  phi_small    phi - phi_grand: everything the modality earns in SMALLER coalitions.",
            "  dagger       |SNR| = |mean/std over folds| < 2 -> effect not resolved.",
            "=" * 78, ""]
    for i, k in enumerate(allrows, start=1):
        out += [f"%% ---------- TABLE {i}: {k} ----------", table(k, allrows[k]), ""]
    out += ["%% ---------- OPTIONAL: the models side by side (normalised only) ----------",
            side_by_side(cfg, allrows), ""]
    for k in allrows:
        out += [f"%% ---------- OPTIONAL appendix: {k}, mean +- std over the 5 folds ----------", extended(allrows[k]), ""]
    (C.out(cfg, "tables") / "tables_shapley_ps.txt").write_text("\n".join(out))
