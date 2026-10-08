"""
f1_weighted of all 16 sensor coalitions, CoM and EmbraceNet — the very v(S) the Shapley
values are computed from (EmbraceNet: mean over the R draws), v(∅) = 0.5127.

  tables/table_coalitions_f1.txt            main experiment, both models (LaTeX)
  tables/table_coalitions_f1_noise.txt      one table per model, one column per sigma
  tables/table_coalitions_f1_cloud_gap.txt  one table per model, one column per share of S2 dates lost
  data/coalitions_f1_weighted[_noise|_cloud_gap][_per_fold].csv
"""
import itertools
import pickle

import pandas as pd

from shapfusion import config as C
from shapfusion.analysis import games as G
from shapfusion.shapley.characteristic import V_EMPTY_F1
from shapfusion.shapley.core import coalition_scalar

METRIC = "f1_weighted"
VIEW_ORDER = ["S2_S2VI", "S1", "weather", "DEM"]
VIEW_LABEL = {"S2_S2VI": "S2", "S1": "S1", "weather": "Weather", "DEM": "DEM"}
FLAGS = [VIEW_LABEL[v] for v in VIEW_ORDER]
YES, NO = "✓", "×"
TEX_YES, TEX_NO = r"\checkmark", r"$\times$"

EMPTY_NOTE = (r"The coalition with no modality is the uniform classifier "
              r"$v(\emptyset) = 2p^2/(2p+1) + 2(1-p)^2/(3-2p) = 0.513$ ($p = 0.657$, positive share "
              r"of the test folds), shared by both architectures.")
LEGEND = r"Means over 5 folds. \checkmark: modality active; $\times$: modality absent."

SETTINGS = {
    "main": dict(suffix="", level_col=None),
    "cloud_gap": dict(
        suffix="_cloud_gap", level_col="n_missing", header=r"S2 dates missing (\%)",
        level_tex=lambda k, cfg: f"{round(100 * k / G.seq_len(cfg))}",
        caption=(r"F1-weighted scores for all $2^4 = 16$ modality coalitions under the cloud-gap "
                 r"ablation, %s. At each level, that share of the 12 S2 acquisition dates is removed "
                 r"from every sample and refilled by linear interpolation over the remaining dates, "
                 r"and the model is retrained. " + LEGEND + r" At 100\%% no date is left to "
                 r"interpolate from and S2 is replaced by its per-band training mean. " + EMPTY_NOTE)),
    "noise": dict(
        suffix="_noise", level_col="sigma", header=r"Gaussian noise $\sigma$ on S2",
        level_tex=lambda s, cfg: f"{s:g}",
        caption=(r"F1-weighted scores for all $2^4 = 16$ modality coalitions under Gaussian noise on "
                 r"S2, %s. Noise $\mathcal{N}(0,\sigma^2)$ is added to the standardised S2 input at "
                 r"training and test time, and the model is retrained at every $\sigma$. "
                 + LEGEND + " " + EMPTY_NOTE)),
}


def coalition_order():
    return [frozenset(c) for k in range(len(VIEW_ORDER) + 1) for c in itertools.combinations(VIEW_ORDER, k)]


def levels(cfg, key):
    return [None] if key == "main" else C.levels(cfg, key)


def per_fold(cfg, key):
    rows = []
    for model in G.models(cfg):
        for lv in levels(cfg, key):
            for f in cfg["folds"]:
                vd = pickle.loads((C.job_dir(cfg, model, key, lv, f) / "vdict.pkl").read_bytes())
                recs = vd["coalitions"]
                assert len(recs) == 16, f"{model} {key} {lv} fold {f}: {len(recs)} coalitions"
                for S, rec in recs.items():
                    val = V_EMPTY_F1 if not S else coalition_scalar(rec, METRIC, vd["y_true"])
                    rows.append({"model": model, "level": lv, "fold": f, "coalition": S, METRIC: val})
    return pd.DataFrame(rows)


def flags(S):
    return {VIEW_LABEL[v]: YES if v in S else NO for v in VIEW_ORDER}


def summarise(cfg, df):
    rows = []
    for model in G.models(cfg):
        for level in df[df.model == model].level.unique():
            for S in coalition_order():
                vals = df[(df.model == model) & (df.level.eq(level) if level is not None else df.level.isna())
                          & (df.coalition == S)][METRIC].to_numpy()
                assert len(vals) == len(cfg["folds"]), f"{model} {level} {sorted(S)}: {len(vals)} folds"
                rows.append({"model": model, "level": level, **flags(S), "size": len(S),
                             "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")})
    return pd.DataFrame(rows)


def by_coalition(summ):
    return {(r.model, None if pd.isna(r.level) else r.level,
             frozenset(v for v in VIEW_ORDER if r[VIEW_LABEL[v]] == YES)): r for _, r in summ.iterrows()}


def _rows(lines, cells_for):
    prev = 0
    for S in coalition_order():
        if len(S) != prev:
            lines.append(r"\midrule")
            prev = len(S)
        lines.append(" " + " & ".join([TEX_YES if v in S else TEX_NO for v in VIEW_ORDER] + cells_for(S)) + r" \\")


def latex_main(cfg, summ, with_std=False):
    lookup = by_coalition(summ)

    def cell(model, S):
        r = lookup[(model, None, S)]
        return f"${r['mean']:.3f} \\pm {r['std']:.3f}$" if with_std else f"{r['mean']:.3f}"

    lines = []
    if not with_std:
        lines += [r"\begin{table}[t]", r"\centering",
                  r"\caption{F1-weighted scores for all $2^4 = 16$ modality coalitions, "
                  r"means over 5 folds. \checkmark: modality active; $\times$: modality absent. " + EMPTY_NOTE + "}",
                  r"\label{tab:coalitions_f1}"]
    lines += [r"\begin{tabular}{" + "c" * len(VIEW_ORDER) + "r" * len(G.models(cfg)) + "}", r"\toprule",
              " & ".join(FLAGS + G.models(cfg)) + r" \\", r"\midrule"]
    _rows(lines, lambda S: [cell(m, S) for m in G.models(cfg)])
    lines += [r"\bottomrule", r"\end{tabular}"]
    if not with_std:
        lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_sweep(cfg, key, summ, model):
    st = SETTINGS[key]
    lvls = levels(cfg, key)
    lookup = by_coalition(summ)
    n = len(lvls)
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\caption{" + st["caption"] % model + "}",
             r"\label{tab:coalitions_f1_%s_%s}" % (key, model.lower()),
             r"\begin{tabular}{" + "c" * len(VIEW_ORDER) + "r" * n + "}", r"\toprule",
             "\\multicolumn{%d}{c}{Modality} & \\multicolumn{%d}{c}{%s} \\\\" % (len(VIEW_ORDER), n, st["header"]),
             "\\cmidrule(lr){1-%d}\\cmidrule(lr){%d-%d}" % (len(VIEW_ORDER), len(VIEW_ORDER) + 1, len(VIEW_ORDER) + n),
             " & ".join(FLAGS + [st["level_tex"](lv, cfg) for lv in lvls]) + r" \\", r"\midrule"]
    _rows(lines, lambda S: [f"{lookup[(model, lv, S)]['mean']:.3f}" for lv in lvls])
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def write_setting(cfg, key):
    st = SETTINGS[key]
    df = per_fold(cfg, key)
    summ = summarise(cfg, df)
    sfx, lcol = st["suffix"], st["level_col"]
    data, tables = C.out(cfg, "data"), C.out(cfg, "tables")

    long = df.copy()
    for v in VIEW_ORDER:
        long[VIEW_LABEL[v]] = long.coalition.map(lambda S, v=v: int(v in S))
    long["size"] = long.coalition.map(len)
    cols = ["model"] + ([lcol] if lcol else []) + ["fold", "size"] + FLAGS + [METRIC]
    long = long.rename(columns={"level": lcol} if lcol else {})
    sort = ["model"] + ([lcol] if lcol else []) + ["size"] + FLAGS + ["fold"]
    long.sort_values(sort, ascending=[True] * (len(sort) - len(FLAGS) - 1) + [False] * len(FLAGS) + [True])[cols] \
        .to_csv(data / f"coalitions_f1_weighted{sfx}_per_fold.csv", index=False)

    if lcol is None:
        wide = summ[summ.model == G.models(cfg)[0]][FLAGS + ["size"]].reset_index(drop=True)
        for m in G.models(cfg):
            part = summ[summ.model == m].reset_index(drop=True)
            wide[f"{m}_mean"], wide[f"{m}_std"] = part["mean"], part["std"]
        wide.to_csv(data / "coalitions_f1_weighted.csv", index=False)
        body = ["%% ---------- TABLE: means over 5 folds ----------", latex_main(cfg, summ), "",
                "%% ---------- OPTIONAL appendix: mean +- std over the 5 folds ----------",
                latex_main(cfg, summ, with_std=True)]
    else:
        out = summ.rename(columns={"level": lcol})
        if key == "cloud_gap":
            out.insert(2, "missing_pct", 100 * out[lcol] / G.seq_len(cfg))
        out.to_csv(data / f"coalitions_f1_weighted{sfx}.csv", index=False)
        body = []
        for i, m in enumerate(G.models(cfg), start=1):
            body += [f"%% ---------- TABLE {i}: {m}, means over 5 folds ----------", latex_sweep(cfg, key, summ, m), ""]

    txt = ["=" * 78, f"f1_weighted of all 16 modality coalitions -- {' and '.join(G.models(cfg))} -- setting: {key}",
           "Generated by `python -m shapfusion report --only B`.", "",
           "  v(S) is read from the v-dicts (shapfusion/shapley/core.py: coalition_scalar), i.e. the",
           "  same values the Shapley values are computed from; EmbraceNet: mean over the R draws.",
           "  Mean over the folds; std (CSV) is ddof=1.",
           "  All-x row = v(0) = 0.5127, the uniform classifier (p = 0.65744; same for both models).",
           "  LaTeX needs booktabs (\\toprule, \\cmidrule) and amssymb (\\checkmark).", "=" * 78, ""] + body + [""]
    (tables / f"table_coalitions_f1{sfx}.txt").write_text("\n".join(txt))


def run(cfg):
    for key in SETTINGS:
        write_setting(cfg, key)
