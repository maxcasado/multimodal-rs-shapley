"""
The local (per-sample) analyses, in order: P0 checks (hard gate), then tasks 6-24. Each
task writes its figures / tables to <out>/local_analysis/<section>/ and the numbers it
computed to <section>/key_<task>.json.

    P1 robustness of the maps  (6 cell counts, 7-10 fold stability / E|φ| / shares / clipping,
                                11 regional aggregation)
    P1 model comparison        (12 correlations, 13 difference maps, 14 Moran's I)
    P2 value functions         (15 log-odds, 16 true class)
    P3 missing sensors         (17 accuracy under coalitions, 18 loss without S2 vs interactions)
    P4 hypotheses              (19 weather / DEM as location proxy, 20 S1 as backup sensor)
    P5 interactions            (21 split by context size, 22 DSensD+ agreement term,
                                23 normalised interaction, 24 complementary cells)
"""
import time
import warnings

from . import config, data, p0, p1, p1m, p2, p3, p4, p5
from .config import section

TASKS = [
    ("t06", "p1_robustness", p1.task06_counts),
    ("t07_10", "p1_robustness", p1.task07_10_maps),
    ("t11", "p1_regional", p1.task11_regional),
    ("t12", "p1_models", p1m.task12_correlations),
    ("t13", "p1_models", p1m.task13_differences),
    ("t14", "p1_models", p1m.task14_moran),
    ("t15", "p2_value_function", p2.task15_logodds),
    ("t16", "p2_value_function", p2.task16_true_class),
    ("t17", "p3_missing", p3.task17_coalition_accuracy),
    ("t18", "p3_missing", p3.task18_loss_vs_interactions),
    ("t19", "p4_hypotheses", p4.task19_location_proxy),
    ("t20", "p4_hypotheses", p4.task20_s1_backup),
    ("t21", "p5_interactions", p5.task21_context),
    ("t22", "p5_interactions", p5.task22_agreement),
    ("t23", "p5_interactions", p5.task23_normalised),
    ("t24", "p5_interactions", p5.task24_complementarity),
]


def run(cfg, only=None, log=print) -> dict:
    with warnings.catch_warnings():
        # all-NaN cells / single-fold cells in nanmean / nanstd are expected (masked downstream)
        warnings.simplefilter("ignore", RuntimeWarning)
        return _run(cfg, only, log)


def _run(cfg, only, log) -> dict:
    la = config.build(cfg)
    t0 = time.time()
    log("  P0 checks")
    key = {"p0": p0.run(la)}
    for name, sect, fn in TASKS:
        if only and name not in only:
            continue
        t = time.time()
        k = fn(la)
        data.write_json(k, section(la, sect) / f"key_la_{name}.json")
        key[name] = k
        log(f"  {name} ({sect}) {time.time() - t:.0f}s")
    log(f"  local analysis done in {time.time() - t0:.0f}s -> {la['out_root_path']}")
    return key
