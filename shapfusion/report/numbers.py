"""
Every number the paper quotes, in one place: numbers/report_numbers.{json,md}. From the
v-dicts and the per-fold Perceptual Score, with v(∅) = 0.5127. Each value is the mean ± std
(ddof=1) over the folds, ``unresolved`` = |mean| < 2·std.

    a  the 16 v(S) per model (main experiment), f1 of the full model
    b  per model and modality: phi, phi share (% of Σphi), PS, PS share, phi_grand, phi_small,
       phi_grand/phi (%), v(N) - v(N∖{m}); and the sums
    c  per model and pair: I, I_vide, I_ctx (+ SNR); per modality, Harsanyi dividends summed
       by order, D_k(m) = Σ_{T∋m, |T|=k} d(T), k = 1..4 (phi_m = Σ_k D_k(m)/k)
    d  noise: phi per modality and sigma; phi(S2) loss and weather/S1/DEM gains between the
       first and last sigma; S2/weather crossing sigma (linear interpolation) for phi and PS;
       phi(S2) at sigma = 1, 2, 5
    e  cloud gap: phi(S2) loss from 0 to the last non-degenerate level (11/12 = 92 %), S2/weather
       crossing (% of dates), phi(weather), phi(S1) at 92 %, full-model f1 at 0 and 92 %, every
       value at the degenerate 100 % level
    f  interactions under degradation: I and I_ctx of the 6 pairs at every level

Per-fold ratios (shares, phi_grand/phi, % loss) are computed per fold, then averaged.
Crossings: ``mean_curve`` interpolates the fold-mean curves; mean/std are over the folds whose
own curves cross (``n_folds``).
"""
import itertools
import json
import pickle

import numpy as np
import pandas as pd

from shapfusion import config as C
from shapfusion.analysis import games as G
from shapfusion.analysis.stats import SNR_THRESHOLD
from shapfusion.shapley.characteristic import UNIFORM_P

V = G.VIEW_ORDER
LAB = G.VIEW_LABELS
S2, WEATHER = "S2_S2VI", "weather"


def coalition_name(S) -> str:
    return "+".join(LAB[v] for v in V if v in S) or "∅"


def _crossing(x, a, b):
    """First x where a - b goes from > 0 to <= 0, linearly interpolated; nan if none."""
    d = np.asarray(a, float) - np.asarray(b, float)
    for i in range(len(d) - 1):
        if d[i] > 0 >= d[i + 1]:
            return float(x[i] + (x[i + 1] - x[i]) * d[i] / (d[i] - d[i + 1]))
    return float("nan")


def collect(cfg, games, ps_main, ps_noise) -> dict:
    gm = {(r.model, r.experiment, r.level, r.fold): r.game for r in games.itertuples(index=False)}
    folds = list(cfg["folds"])
    sig = G.sigmas(cfg)
    lv = G.gap_levels(cfg)
    n = G.seq_len(cfg)
    noise_end, gap_end = sig[-1], max(l for l in lv if l < 1.0)
    out = {}

    def put(path, vals):
        out[tuple(path)] = np.asarray(vals, float)

    coal = [frozenset(c) for k in range(len(V) + 1) for c in itertools.combinations(V, k)]
    full = frozenset(V)
    for M in G.models(cfg):
        main = [gm[(M, "main", 0.0, f)] for f in folds]
        for S in coal:                                                      # a
            put(("a", M, coalition_name(S)), [g[S] for g in main])
        put(("a", M, "f1_full_model"), [g[full] for g in main])

        dec = [G.decomposition(g) for g in main]                            # b
        ps = {m: [float(ps_main[(ps_main.model == M) & (ps_main.fold == f) & (ps_main.modality == m)].PS_raw.iloc[0])
                  for f in folds] for m in V}
        phi_tot = np.array([sum(d[m]["phi"] for m in V) for d in dec])
        ps_tot = np.sum([ps[m] for m in V], axis=0)
        for m in V:
            phi = np.array([d[m]["phi"] for d in dec])
            grand = np.array([d[m]["phi_grand"] for d in dec])
            b = ("b", M, LAB[m])
            put(b + ("phi",), phi)
            put(b + ("phi_share_pct",), 100 * phi / phi_tot)
            put(b + ("PS",), ps[m])
            put(b + ("PS_share_pct",), 100 * np.array(ps[m]) / ps_tot)
            put(b + ("phi_grand",), grand)
            put(b + ("phi_small",), phi - grand)
            put(b + ("phi_grand_over_phi_pct",), 100 * grand / phi)
            put(b + ("marginal_grand",), [d[m]["marginal_grand"] for d in dec])
        sm = ("b", M, "Sum")
        grand_tot = np.array([sum(d[m]["phi_grand"] for m in V) for d in dec])
        put(sm + ("phi",), phi_tot)
        put(sm + ("phi_share_pct",), np.full(len(folds), 100.0))
        put(sm + ("PS",), ps_tot)
        put(sm + ("PS_share_pct",), np.full(len(folds), 100.0))
        put(sm + ("phi_grand",), grand_tot)
        put(sm + ("phi_small",), phi_tot - grand_tot)
        put(sm + ("phi_grand_over_phi_pct",), 100 * grand_tot / phi_tot)
        put(sm + ("marginal_grand",), [sum(d[m]["marginal_grand"] for m in V) for d in dec])

        split = [G.interaction_split(g) for g in main]                      # c
        for pair in G.PAIRS:
            for j, q in enumerate(("I", "I_vide", "I_ctx")):
                put(("c", M, "pairs", G.pair_label(pair), q), [s[pair][j] for s in split])
        div = [G.harsanyi(g) for g in main]
        for m in V:
            for k in range(1, len(V) + 1):
                put(("c", M, "harsanyi_by_order", LAB[m], str(k)),
                    [sum(x for T, x in d.items() if m in T and len(T) == k) for d in div])

        phi_n = {s: [G.phi(gm[(M, "noise", s, f)]) for f in folds] for s in sig}       # d
        for s in sig:
            for m in V:
                put(("d", M, "phi", f"sigma={s:g}", LAB[m]), [p[m] for p in phi_n[s]])
        put(("d", M, "S2_phi_loss_first_to_last"), [phi_n[sig[0]][i][S2] - phi_n[noise_end][i][S2] for i in range(len(folds))])
        put(("d", M, "S2_phi_loss_first_to_last_pct"),
            [100 * (phi_n[sig[0]][i][S2] - phi_n[noise_end][i][S2]) / phi_n[sig[0]][i][S2] for i in range(len(folds))])
        for m in V[1:]:
            put(("d", M, "gain_first_to_last", LAB[m]),
                [phi_n[noise_end][i][m] - phi_n[sig[0]][i][m] for i in range(len(folds))])
        for s in (1.0, 2.0, 5.0):
            if s in phi_n:
                put(("d", M, "phi_S2_at", f"sigma={s:g}"), [p[S2] for p in phi_n[s]])
        psn = ps_noise[ps_noise.model == M]
        ps_at = {(s, f, m): float(psn[(psn.level == s) & (psn.fold == f) & (psn.modality == m)].PS_raw.iloc[0])
                 for s in sig for f in folds for m in V}
        put(("d", M, "crossing_S2_weather_sigma", "phi"),
            [_crossing(sig, [phi_n[s][i][S2] for s in sig], [phi_n[s][i][WEATHER] for s in sig]) for i in range(len(folds))])
        put(("d", M, "crossing_S2_weather_sigma", "PS"),
            [_crossing(sig, [ps_at[(s, f, S2)] for s in sig], [ps_at[(s, f, WEATHER)] for s in sig]) for f in folds])

        phi_g = {l: [G.phi(gm[(M, "cloud_gap", l, f)]) for f in folds] for l in lv}     # e
        p0, pend = phi_g[lv[0]], phi_g[gap_end]
        put(("e", M, "S2_phi_loss_0_to_last"), [p0[i][S2] - pend[i][S2] for i in range(len(folds))])
        put(("e", M, "S2_phi_loss_0_to_last_pct"), [100 * (p0[i][S2] - pend[i][S2]) / p0[i][S2] for i in range(len(folds))])
        nd = [l for l in lv if l <= gap_end]
        put(("e", M, "crossing_S2_weather_pct_dates"),
            [100 * _crossing(nd, [phi_g[l][i][S2] for l in nd], [phi_g[l][i][WEATHER] for l in nd]) for i in range(len(folds))])
        put(("e", M, "phi_at_last", "weather"), [p[WEATHER] for p in pend])
        put(("e", M, "phi_at_last", "S1"), [p["S1"] for p in pend])
        put(("e", M, "f1_full_model", "0%"), [gm[(M, "cloud_gap", lv[0], f)][full] for f in folds])
        put(("e", M, "f1_full_model", f"{gap_end:.0%}"), [gm[(M, "cloud_gap", gap_end, f)][full] for f in folds])
        if 1.0 in phi_g:
            for m in V:
                put(("e", M, "at_100pct", f"phi_{LAB[m]}"), [p[m] for p in phi_g[1.0]])
            put(("e", M, "at_100pct", "f1_full_model"), [gm[(M, "cloud_gap", 1.0, f)][full] for f in folds])

        for exp, levels in (("noise", sig), ("cloud_gap", lv)):             # f
            for l in levels:
                lab = f"sigma={l:g}" if exp == "noise" else f"{round(l * n)}/{n} ({l:.0%})"
                sp = [G.interaction_split(gm[(M, exp, l, f)]) for f in folds]
                for pair in G.PAIRS:
                    put(("f", M, exp, lab, G.pair_label(pair), "I"), [s[pair][0] for s in sp])
                    put(("f", M, exp, lab, G.pair_label(pair), "I_ctx"), [s[pair][2] for s in sp])
    return out


def mean_curve_crossings(cfg, games, ps_noise) -> dict:
    from collections import defaultdict
    acc = defaultdict(list)
    for r in games.itertuples(index=False):
        if r.experiment in ("noise", "cloud_gap"):
            p = G.phi(r.game)
            acc[(r.model, r.experiment, r.level)].append((p[S2], p[WEATHER]))
    out = {}
    for M in G.models(cfg):
        for exp, levels in (("noise", G.sigmas(cfg)), ("cloud_gap", [l for l in G.gap_levels(cfg) if l < 1.0])):
            a = [np.mean([x[0] for x in acc[(M, exp, l)]]) for l in levels]
            b = [np.mean([x[1] for x in acc[(M, exp, l)]]) for l in levels]
            c = _crossing(levels, a, b)
            out[(M, exp, "phi")] = c * 100 if exp == "cloud_gap" else c
        p = ps_noise[ps_noise.model == M].groupby(["level", "modality"]).PS_raw.mean()
        out[(M, "noise", "PS")] = _crossing(G.sigmas(cfg), [p[(s, S2)] for s in G.sigmas(cfg)],
                                            [p[(s, WEATHER)] for s in G.sigmas(cfg)])
    return out


def leaf(vals) -> dict:
    v = np.asarray(vals, float)
    n_nan = int(np.isnan(v).sum())
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"mean": None, "std": None, "snr": None, "unresolved": True, "n_folds": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    snr = mean / std if std > 0 else (float("inf") if mean else float("nan"))
    out = {"mean": mean, "std": std, "snr": snr if np.isfinite(snr) else None,
           "unresolved": bool(abs(mean) < SNR_THRESHOLD * std)}
    if n_nan:
        out["n_folds"] = int(v.size)
    return out


def _finite(o):
    """NaN / inf -> None (strict JSON)."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def nest(flat: dict) -> dict:
    root = {}
    for path, val in flat.items():
        d = root
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = val
    return root


# ── markdown ─────────────────────────────────────────────────────────────────
def _v(x, pct=False, dec=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.1f}" if pct else f"{x:.{dec}f}"


def _cell(l, n_folds, pct=False, dec=3):
    if l["mean"] is None:
        return "—"
    s = f"{_v(l['mean'], pct, dec)} ± {_v(l['std'], pct, dec)}"
    if l.get("n_folds") is not None:
        s += f" ({l['n_folds']}/{n_folds} folds)"
    return s + (" · n.r." if l["unresolved"] else "")


def markdown(cfg, R: dict, meta: dict, cross: dict) -> str:
    nf = len(cfg["folds"])
    head = f"| | mean ± std ({nf} folds) |\n|---|---|"

    def row(name, l, pct=False, dec=3):
        return f"| {name} | {_cell(l, nf, pct, dec)} |"

    L = ["# Numbers of the paper", "",
         f"v(∅) = **{meta['v_empty']:.4f}** everywhere (uniform classifier, 2p²/(2p+1) + 2(1−p)²/(3−2p), "
         f"p = {meta['p_constant']}; positive share of the test folds of this run: {meta['p']}, n = {meta['n']}). "
         f"Mean ± std (ddof = 1) over {nf} folds; **n.r.** = not resolved, |mean| < 2 std.", ""]
    for M in G.models(cfg):
        L += [f"## {M}", "", "### a. The 16 coalitions v(S) (f1_weighted)", "", head]
        for k, l in R["a"][M].items():
            L.append(row("**f1 of the full model**" if k == "f1_full_model" else k, l))
        L += ["", "### b. Attribution per modality", ""]
        for key, name, pct in [("phi", "φ", False), ("phi_share_pct", "φ (% of Σφ)", True), ("PS", "PS", False),
                               ("PS_share_pct", "PS (% of ΣPS)", True), ("phi_grand", "φ_grand", False),
                               ("phi_small", "φ_small", False), ("phi_grand_over_phi_pct", "φ_grand/φ (%)", True),
                               ("marginal_grand", "v(N) − v(N∖m)", False)]:
            L += [f"**{name}**", "", head] + [row(mod, d[key], pct) for mod, d in R["b"][M].items()] + [""]
        L += ["### c. Interactions (main experiment)", "",
              "| pair | I | SNR | I_vide | SNR | I_ctx | SNR |", "|---|---|---|---|---|---|---|"]
        snr = lambda l: "—" if l["snr"] is None else f"{l['snr']:+.1f}"  # noqa: E731
        for p, d in R["c"][M]["pairs"].items():
            L.append(f"| {p} | {_cell(d['I'], nf)} | {snr(d['I'])} | {_cell(d['I_vide'], nf)} | {snr(d['I_vide'])} "
                     f"| {_cell(d['I_ctx'], nf)} | {snr(d['I_ctx'])} |")
        L += ["", "Harsanyi dividends summed by order, D_k(m) = Σ_{T∋m, |T|=k} d(T) (φ_m = Σ_k D_k(m)/k):", "",
              "| modality | order 1 | order 2 | order 3 | order 4 |", "|---|---|---|---|---|"]
        for mod, d in R["c"][M]["harsanyi_by_order"].items():
            L.append(f"| {mod} | " + " | ".join(_cell(d[k], nf) for k in "1234") + " |")
        dd = R["d"][M]
        L += ["", "### d. Gaussian noise on S2", "", "| σ | " + " | ".join(LAB[m] for m in V) + " |",
              "|---" * (len(V) + 1) + "|"]
        for s, d in dd["phi"].items():
            L.append(f"| {s} | " + " | ".join(_cell(d[LAB[m]], nf) for m in V) + " |")
        L += ["", "(φ)", "", head, row("loss of φ(S2), first → last σ", dd["S2_phi_loss_first_to_last"]),
              row("loss of φ(S2), first → last σ (%)", dd["S2_phi_loss_first_to_last_pct"], True)]
        for mod, l in dd["gain_first_to_last"].items():
            L.append(row(f"gain of φ({mod}), first → last σ", l))
        for q in ("phi", "PS"):
            L.append(row(f"S2/weather crossing σ, {q} (per fold)", dd["crossing_S2_weather_sigma"][q], dec=2))
            L.append(f"| S2/weather crossing σ, {q} (mean curves) | {_v(cross[(M, 'noise', q)], dec=2)} |")
        for s, l in dd.get("phi_S2_at", {}).items():
            L.append(row(f"φ(S2) at {s}", l, dec=4))
        de = R["e"][M]
        L += ["", "### e. Cloud gaps (S2 dates removed)", "", head,
              row("loss of φ(S2), 0 → last level", de["S2_phi_loss_0_to_last"]),
              row("loss of φ(S2), 0 → last level (%)", de["S2_phi_loss_0_to_last_pct"], True),
              row("S2/weather crossing, % of dates (per fold)", de["crossing_S2_weather_pct_dates"], True),
              f"| S2/weather crossing, % of dates (mean curves) | {_v(cross[(M, 'cloud_gap', 'phi')], True)} |",
              row("φ(weather) at the last level", de["phi_at_last"]["weather"]),
              row("φ(S1) at the last level", de["phi_at_last"]["S1"])]
        for k, l in de["f1_full_model"].items():
            L.append(row(f"f1 of the full model at {k}", l))
        for k, l in de.get("at_100pct", {}).items():
            L.append(row(f"100 % (degenerate): {k}", l))
        L += ["", "### f. Interactions under degradation (I | I_ctx)", ""]
        for exp, name in (("noise", "noise"), ("cloud_gap", "cloud gaps")):
            pairs = [G.pair_label(p) for p in G.PAIRS]
            L += [f"**{name}**", "", "| level | " + " | ".join(pairs) + " |", "|---" * (len(pairs) + 1) + "|"]
            for lab, d in R["f"][M][exp].items():
                L.append(f"| {lab} | " + " | ".join(f"{_cell(d[p]['I'], nf)} \\| {_cell(d[p]['I_ctx'], nf)}"
                                                     for p in pairs) + " |")
            L.append("")
        L.append("")
    return "\n".join(L)


def run(cfg):
    games = G.load_games(cfg)
    ps_main = pd.concat([G.load_ps(cfg, M, "main") for M in G.models(cfg)], ignore_index=True)
    ps_noise = pd.concat([G.load_ps(cfg, M, "noise") for M in G.models(cfg)], ignore_index=True)
    flat = collect(cfg, games, ps_main, ps_noise)
    R = nest({k: leaf(v) for k, v in flat.items()})
    cross = mean_curve_crossings(cfg, games, ps_noise)
    for M in G.models(cfg):
        for q in ("phi", "PS"):
            R["d"][M]["crossing_S2_weather_sigma"][q]["mean_curve"] = cross[(M, "noise", q)]
        R["e"][M]["crossing_S2_weather_pct_dates"]["mean_curve"] = cross[(M, "cloud_gap", "phi")]
    ys = [np.asarray(pickle.loads((d / "vdict.pkl").read_bytes())["y_true"]).ravel()
          for _, dirs in G.sources(cfg, G.models(cfg)[0], "main") for d in dirs.values()]
    y = np.concatenate(ys)
    meta = {"v_empty": G.V_EMPTY_F1, "p_constant": UNIFORM_P, "p": round(float(y.mean()), 5), "n": int(y.size),
            "p_per_fold": [round(float(v.mean()), 4) for v in ys],
            "rule": "unresolved = |mean| < 2*std (ddof=1 over the folds)"}
    names = {"a": "a_coalitions", "b": "b_attribution", "c": "c_interactions", "d": "d_noise",
             "e": "e_cloud_gap", "f": "f_interactions_degradation"}
    doc = {"meta": meta, **{names[k]: R[k] for k in "abcdef"}}
    out = C.out(cfg, "numbers")
    (out / "report_numbers.json").write_text(json.dumps(_finite(doc), indent=2, ensure_ascii=False,
                                                        allow_nan=False))
    (out / "report_numbers.md").write_text(markdown(cfg, R, meta, cross))
