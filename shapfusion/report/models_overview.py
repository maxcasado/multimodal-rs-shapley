"""
The four models side by side:
  tables/table_metrics_global.txt, data/metrics_global{,_per_fold}.csv
      accuracy and f1_weighted of the FULL coalition on each test fold, right after training
      (one forward pass; for EmbraceNet one stochastic draw), mean ± std over the folds
  figures/models_overview/fig_shapley_local_map[_std]_<model>.png           local phi, mean / within-cell std
  figures/models_overview/fig_shapley_local_interactions[_std]_<model>.png  local pairwise interactions
Figures of the same kind share ONE colour scale across the models (98th percentile of
|cell value| pooled over every model's panels).
"""
import json

import pandas as pd

from shapfusion import config as C
from shapfusion.analysis import local_maps as lm
from shapfusion.local_analysis import config as la_config, games as LG

METRICS = ["accuracy", "f1_weighted"]


def models(cfg) -> dict:
    return dict(cfg["report"]["overview"])


def metrics_tables(cfg):
    rows = []
    for m, name in models(cfg).items():
        for f in cfg["folds"]:
            r = json.loads((C.job_dir(cfg, m, "main", None, f) / "metrics.json").read_text())
            rows.append({"model": name, "fold": f, **{k: float(r[k]) for k in METRICS}, "epochs": int(r["epochs"])})
    per_fold = pd.DataFrame(rows)
    data = C.out(cfg, "data")
    per_fold.to_csv(data / "metrics_global_per_fold.csv", index=False)
    summ = per_fold.groupby("model", sort=False)[METRICS].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ["n_folds"] = per_fold.groupby("model", sort=False).size()
    summ.to_csv(data / "metrics_global.csv")
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Model & Accuracy & F1\textsubscript{weighted} \\", r"\midrule"]
    for name, r in summ.iterrows():
        lines.append(f"{name} & {r.accuracy_mean:.3f} $\\pm$ {r.accuracy_std:.3f} & "
                     f"{r.f1_weighted_mean:.3f} $\\pm$ {r.f1_weighted_std:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (C.out(cfg, "tables") / "table_metrics_global.txt").write_text("\n".join(lines) + "\n")


def local_maps(cfg):
    la = la_config.build(cfg)
    dfs = {m: LG.sample_frame(la, m, "prob_pred", with_regions=False) for m in models(cfg)}
    figs = C.out(cfg, "figures") / "models_overview"
    kinds = [("shapley_local_map", [lm.phi_col(v) for v in lm.ORDER], lm.fig_local_shapley_maps),
             ("shapley_local_interactions", [lm.sii_col(a, b) for a, b in lm.PAIRS], lm.fig_local_interaction_maps)]
    for stem, cols, draw in kinds:
        for stat, suffix in (("mean", ""), ("std", "_std")):
            vmax = lm.shared_vmax([lm.grids(df, cols, stat=stat) for df in dfs.values()])
            for m, df in dfs.items():
                draw(df, figs / f"fig_{stem}{suffix}_{m}.png", model_label=models(cfg)[m], stat=stat, vmax=vmax)


def run(cfg):
    metrics_tables(cfg)
    local_maps(cfg)
