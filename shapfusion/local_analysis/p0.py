"""
P0 — correctness checks on the per-sample games of the four models (written to
<out>/local_analysis/p0_checks/; the efficiency and closed-form criteria are hard gates).

1. efficiency      max_x |Σ_i φ_i(x) - (v(N, x) - v(∅, x))| per model × fold × value function.
3. closed form     DSensD+ fuses by the MEAN of the branch logits, so in log-odds
                   v(S) = mean_{k∈S} ℓ_k, v(∅) = b and
                       Δ_ij(∅) = b - (ℓ_i + ℓ_j)/2
                       Δ_ij(S) = -[(ℓ_i - m_S) + (ℓ_j - m_S)] / ((s+1)(s+2)),  S ≠ ∅
                   I_ij from these (branch log-odds only) vs I_ij from shapley/core.py on the
                   coalition log-odds the model produced. Run on the float64 re-inference
                   (logit64) and, for information, on the float32 v-dict outputs.
5. quantisation    distinct values of p(ŷ | x) on the test fold, exact and rounded.
"""
import itertools
import math

import numpy as np
import pandas as pd

from . import games, vcache


def _logit(p):
    return math.log(p) - math.log1p(-p)


# ── 1. efficiency ─────────────────────────────────────────────────────────────

def efficiency(cfg, value_functions=("prob_pred", "logodds_pred", "prob_true")):
    views = cfg["views"]
    rows = []
    for model in cfg["models"]:
        for fold in cfg["folds"]:
            c = vcache.load(cfg, model, fold)
            for vf in value_functions:
                T = games.value_table(c, views, vf, cfg["v_empty"])
                P = games.phi(T, views)
                err = np.abs(P.sum(1) - (T[games.full_mask(views)] - T[0]))
                rows.append({"model": model, "fold": fold, "value_function": vf,
                             "source": "recomputed", "n": len(err),
                             "max_abs_err": float(np.nanmax(err)),
                             "n_nan": int(np.isnan(err).sum())})
    return pd.DataFrame(rows)


# ── 3. closed form for mean-of-logits fusion ──────────────────────────────────

def closed_form_sii(l, views, b):
    """I_ij from the branch log-odds only. ``l``: {view: (N,)}; returns {pair: (N,)}."""
    n = len(views)
    w = {s: math.factorial(s) * math.factorial(n - s - 2) / math.factorial(n - 1)
         for s in range(n - 1)}
    out = {}
    for i, j in itertools.combinations(views, 2):
        others = [v for v in views if v not in (i, j)]
        tot = w[0] * (b - (l[i] + l[j]) / 2.0)
        for s in range(1, len(others) + 1):
            for S in itertools.combinations(others, s):
                m = np.mean([l[k] for k in S], axis=0)
                tot = tot + w[s] * (-((l[i] - m) + (l[j] - m)) / ((s + 1) * (s + 2)))
        out[(i, j)] = tot
    return out


def closed_form(cfg, models=None, keys=("logit64", "logit")):
    views = cfg["views"]
    b = _logit(float(cfg["v_empty"]))
    rows = []
    for model in models or cfg["closed_form_models"]:
        for fold in cfg["folds"]:
            c = vcache.load(cfg, model, fold)
            for key in keys:
                if key not in c:
                    continue
                # oriented on ŷ_full like every game here: v(S) = mean_k (±ℓ_k) is still a
                # mean of branch values, so the closed form applies to the oriented ℓ
                sign = np.where(games.yhat_full(c, views) == 1, 1.0, -1.0)
                T = sign * c[key]
                T[0] = b
                I_sh = games.sii(T, views)
                l = {v: T[1 << k] for k, v in enumerate(views)}
                cf = closed_form_sii(l, views, b)
                I_cf = np.stack([cf[p] for p in games.pairs(views)], axis=1)
                # linearity of the fusion itself: coalition log-odds = mean of branch log-odds
                lin = max(np.abs(T[m] - np.mean([T[1 << k] for k in range(len(views)) if m >> k & 1],
                                                 axis=0)).max() for m in range(1, 16))
                rows.append({"model": model, "fold": fold, "logits": key,
                             "max_abs_err": float(np.abs(I_sh - I_cf).max()),
                             "max_abs_I": float(np.abs(I_sh).max()),
                             "max_abs_branch_logodds": float(max(np.abs(x).max() for x in l.values())),
                             "max_fusion_linearity_err": float(lin)})
    return pd.DataFrame(rows)


# ── 5. quantisation of p(ŷ | x) ───────────────────────────────────────────────

def quantisation(cfg, model):
    views = cfg["views"]
    step = float(cfg["tolerances"]["quantisation_round"])
    dec = int(round(-math.log10(step)))
    rows = []
    for fold in cfg["folds"]:
        c = vcache.load(cfg, model, fold)
        p = c["p1"][games.full_mask(views)]
        q = np.maximum(p, 1 - p)                       # p(ŷ_full | x)
        r = np.round(q, dec)
        _, counts = np.unique(r, return_counts=True)
        top = np.sort(counts)[::-1]
        rows.append({"model": model, "fold": fold, "n": len(q),
                     "distinct_exact": int(len(np.unique(q))),
                     f"distinct_round_{step:g}": int(len(counts)),
                     "share_top16_values": float(top[:16].sum() / len(q)),
                     "n_values_for_99pct": int(np.searchsorted(np.cumsum(top) / len(q), 0.99) + 1),
                     **{f"distinct_branch_{games.short(v)}": int(len(np.unique(np.round(
                         c["p1"][1 << k], dec)))) for k, v in enumerate(views)}})
    return pd.DataFrame(rows)


# ── 4. VALUE_FUNCTION.md ──────────────────────────────────────────────────────

def value_function_md(cfg) -> str:
    v0 = float(cfg["v_empty"])
    b = _logit(v0)
    lines = [
        "# Value functions of the local (per-sample) Shapley analyses",
        "",
        "Written by `python -m shapfusion report` (`shapfusion/local_analysis/p0.py`). Applies to "
        "every figure under `local_analysis/p*` and to the local maps of `figures/`.",
        "",
        "## Per-sample game",
        "",
        "Players: the four sensors S2 (S2_S2VI), S1, weather, DEM. For a test sample x and a "
        "coalition S ≠ ∅, the model is run with only the sensors of S (absent branches are dropped "
        "from the fusion, nothing is imputed). ŷ_full is the class the full coalition predicts: "
        "1[P(class 1 | all four) > 0.5]. Every test sample is explained once, by the model of the "
        "fold it is a test sample of.",
        "",
        "| name | v(S), S ≠ ∅ | v(∅) | used for |",
        "|---|---|---|---|",
        f"| prob_pred | P(ŷ_full \\| x_S) | {v0:g} | every map unless stated (the paper's local maps) |",
        f"| logodds_pred | log P(ŷ_full \\| x_S) − log P(other \\| x_S) | b = logit({v0:g}) = {b:g} | P0.3, task 15, task 22 |",
        f"| prob_true | P(y_true \\| x_S) | {v0:g} | task 16 |",
        "",
        f"v(∅) = {v0:g} for every model: with no sensor the model cannot be run, so v(∅) is a "
        "prior, not a model output. The uniform prior is symmetric in the two classes, which makes "
        "orienting the game on ŷ_full an exact sign flip of φ and I, and puts b = 0 in log-odds. "
        "It is NOT the v(∅) of the global tables (f1_weighted of the uniform classifier, 0.5127): "
        "the local φ do not average to the table φ.",
        "",
        "## Sources of v(S) (no model retrained)",
        "",
    ]
    for model, spec in cfg["models"].items():
        if spec.get("draws", 1) > 1:
            s = (f"mean over the R = {spec['draws']} stochastic draws of the v-dict of P(class | x_S) "
                 "(the same draws as the global coalition values)")
        else:
            s = ("softmax output of every coalition stored in the v-dict (float32; on GPU the "
                 "convolutions use TF32, so logits differ from a float64 run by up to ~1e-2)")
        if spec.get("logit64"):
            s += "; plus a float64 re-inference of the raw logits for the closed-form check"
        lines.append(f"- **{spec['label']}**: {s}.")
    if any(s.get("quantised") for s in cfg["models"].values()):
        lines += ["", "## Quantised model", "",
                  "DSensD+ (LN) ends every branch with LayerNorm over C = 2 logits, which maps "
                  "almost every input to one of two vectors ±(γ) + β. Its per-sample games "
                  "therefore take few distinct values (see `p0_checks/quantisation.csv`); "
                  "figures flag it in the row label."]
    return "\n".join(lines) + "\n"


def run(cfg) -> dict:
    """Write p0_checks/ and VALUE_FUNCTION.md; raise if a hard criterion fails."""
    from . import data
    from .config import section
    out = section(cfg, "p0_checks")
    out.mkdir(parents=True, exist_ok=True)
    tol_eff = float(cfg["tolerances"]["efficiency"])
    tol_cf = float(cfg["tolerances"]["closed_form"])
    eff = efficiency(cfg)
    data.write_table(eff, out / "efficiency.csv")
    cf = closed_form(cfg)
    data.write_table(cf, out / "closed_form.csv")
    quant = [quantisation(cfg, m) for m, s in cfg["models"].items() if s.get("quantised")]
    if quant:
        data.write_table(pd.concat(quant), out / "quantisation.csv")
    (cfg["out_root_path"] / "VALUE_FUNCTION.md").write_text(value_function_md(cfg))
    bad_eff = eff[(eff.max_abs_err >= tol_eff) | (eff.n_nan > 0)]
    cf64 = cf[cf.logits == "logit64"] if len(cf) else cf
    bad_cf = cf64[cf64.max_abs_err >= tol_cf] if len(cf64) else cf64
    if not bad_eff.empty or not bad_cf.empty:
        raise AssertionError("P0 checks failed:\n" + bad_eff.to_string() + "\n" + bad_cf.to_string())
    key = {"efficiency_max_abs_err": float(eff.max_abs_err.max()),
           "closed_form_float64_max_abs_err": float(cf64.max_abs_err.max()) if len(cf64) else None,
           "closed_form_float32_vdict_max_abs_err":
               cf[cf.logits == "logit"].groupby("model").max_abs_err.max().to_dict() if len(cf) else {}}
    data.write_json(key, out / "p0_passed.json")
    return key
