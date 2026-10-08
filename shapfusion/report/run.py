"""
Every deliverable of the paper from the artefacts in runs/ (CPU only, no model is run):

    A  main figures            figures/fig_noise_*, fig_phi_decomposition, fig_cloud_gap_*, fig_shapley_local_map_*
    B  coalition tables        tables/table_coalitions_f1{,_noise,_cloud_gap}.txt
    C  Shapley vs PS tables    tables/tables_shapley_ps.txt
    D  global interactions     figures/interactions_degradation.pdf, data/interactions_global*.csv
    E  local interactions      figures/interactions_local_*.pdf, numbers/interactions_local.md
    F  numbers of the text     numbers/report_numbers.{json,md}
    G  the four models         tables/table_metrics_global.txt, figures/models_overview/
    H  local analyses          local_analysis/ (P0 checks, P1-P5)
"""
import time

from shapfusion import config as C
from shapfusion.analysis import games as G

DELIVERABLES = {
    "A": ("main figures", "fig_final"),
    "B": ("coalition tables", "table_coalitions"),
    "C": ("Shapley vs Perceptual Score tables", "table_shapley_ps"),
    "D": ("global interactions", "interactions_global"),
    "E": ("local interactions", "interactions_local"),
    "F": ("numbers of the text", "numbers"),
    "G": ("the four models", "models_overview"),
    "H": ("local analyses", None),
}


def _files(cfg, models, experiments, names):
    return [C.job_dir(cfg, m, e, lv, f) / n for m in models for e in experiments if m in cfg["experiments"][e]["models"]
            for lv in C.levels(cfg, e) for f in cfg["folds"] for n in names]


def inputs(cfg, key) -> list:
    gm = G.models(cfg)
    sweeps = ("main", "noise", "cloud_gap")
    return {"A": _files(cfg, gm, sweeps, ["vdict.pkl"]) + _files(cfg, gm, ["noise"], ["ps.csv"])
                 + _files(cfg, gm, ["main"], ["local.npz"]),
            "B": _files(cfg, gm, sweeps, ["vdict.pkl"]),
            "C": _files(cfg, gm, ["main"], ["vdict.pkl", "ps.csv"]),
            "D": _files(cfg, gm, sweeps, ["vdict.pkl"]),
            "E": _files(cfg, gm, ["main"], ["local.npz"]),
            "F": _files(cfg, gm, sweeps, ["vdict.pkl"]) + _files(cfg, gm, ["main", "noise"], ["ps.csv"]),
            "G": _files(cfg, list(cfg["report"]["overview"]), ["main"], ["metrics.json", "local.npz"]),
            "H": _files(cfg, list(cfg["local_analysis"]["models"]), ["main"], ["vdict.pkl", "local.npz"])}[key]


def run(cfg, only=None, log=print) -> dict:
    status = {}
    for key, (name, module) in DELIVERABLES.items():
        if only and key not in only:
            continue
        missing = [p for p in inputs(cfg, key) if not p.exists()]
        if missing:
            log(f"[{key}] {name}: SKIPPED, {len(missing)} input file(s) missing, e.g. "
                f"{missing[0].relative_to(cfg['root'])} (run `python -m shapfusion run` first)")
            status[key] = "missing inputs"
            continue
        t0 = time.time()
        log(f"[{key}] {name}")
        if module is None:
            from shapfusion.local_analysis import run as la_run
            la_run.run(cfg, log=log)
        else:
            import importlib
            importlib.import_module(f"shapfusion.report.{module}").run(cfg)
        status[key] = f"done in {time.time() - t0:.0f}s"
        log(f"[{key}] {status[key]}")
    log(f"outputs in {cfg['out_path']}")
    return status
