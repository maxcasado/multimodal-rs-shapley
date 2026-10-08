"""
The GLOBAL (fold-level) games of the paper, read from the v-dicts, with ONE empty coalition:
v(∅) = V_EMPTY_F1 = 0.5127 (uniform classifier, shapley/characteristic.py); v(S), S ≠ ∅, is
the f1_weighted of the stored coalition predictions (EmbraceNet: mean over the R draws).

Experiments and levels (shapfusion/config.py): main (level 0), noise (level = sigma),
cloud_gap (level = share k / 12 of the S2 dates lost).

Pure functions on a scalar game {frozenset -> v}:
    harsanyi(v)            15 dividends d(T) = Σ_{S⊆T} (-1)^{|T|-|S|} v(S), T ≠ ∅
    interaction_split(v)   {pair: (I, I_vide, I_ctx)}: I the order-2 Shapley interaction
                           index, I_vide = (1/3)[v(ij) - v(i) - v(j) + v(∅)] its |S| = 0 term
    decomposition(v)       phi = phi_grand + phi_small, phi_grand = [v(N) - v(N∖m)] / n
"""
from __future__ import annotations

import itertools
import pickle

import pandas as pd

from shapfusion import config as C
from shapfusion.shapley.characteristic import DEFAULT_METRIC, V_EMPTY_F1
from shapfusion.shapley.core import scalar_vdict, shapley_interaction_terms, shapley_values
from .plotstyle import VIEW_LABELS, VIEW_ORDER

EXPERIMENTS = ("main", "noise", "cloud_gap")
# The six pairs in the paper's reading order (strongest redundancy first).
PAIRS = [("S2_S2VI", "weather"), ("S2_S2VI", "S1"), ("S1", "weather"),
         ("S2_S2VI", "DEM"), ("S1", "DEM"), ("weather", "DEM")]


def pair_label(pair) -> str:
    return "-".join(VIEW_LABELS[v] for v in pair)


def models(cfg) -> list:
    return list(cfg["report"]["global_models"])


def sigmas(cfg) -> list:
    return [float(s) for s in cfg["experiments"]["noise"]["levels"]]


def gap_levels(cfg) -> list:
    """Cloud-gap levels as shares of the S2 dates lost."""
    return [C.level_value(cfg, "cloud_gap", k) for k in cfg["experiments"]["cloud_gap"]["levels"]]


def seq_len(cfg) -> int:
    return int(cfg["experiments"]["cloud_gap"]["seq_len"])


def sources(cfg, model: str, experiment: str) -> list:
    """[(level value, {fold: job dir})] in sweep order."""
    return [(C.level_value(cfg, experiment, lv),
             {f: C.job_dir(cfg, model, experiment, lv, f) for f in cfg["folds"]})
            for lv in C.levels(cfg, experiment)]


def vdict_paths(cfg, models_=None, experiments=EXPERIMENTS) -> list:
    return [d / "vdict.pkl" for m in (models_ or models(cfg)) for e in experiments
            for _, dirs in sources(cfg, m, e) for d in dirs.values()]


def _coalitions():
    return [frozenset(c) for r in range(len(VIEW_ORDER) + 1) for c in itertools.combinations(VIEW_ORDER, r)]


def load_game(path, v_empty: float = V_EMPTY_F1, metric: str = DEFAULT_METRIC) -> dict:
    """Scalar game {frozenset -> v} of one fold, with the paper's v(∅)."""
    with open(path, "rb") as fh:
        v = scalar_vdict(pickle.load(fh), metric)
    v[frozenset()] = float(v_empty)
    missing = [S for S in _coalitions() if S not in v]
    if missing:
        raise KeyError(f"{path}: coalitions missing {missing}")
    return v


def load_games(cfg, models_=None, experiments=EXPERIMENTS, v_empty: float = V_EMPTY_F1) -> pd.DataFrame:
    """Long table: model, experiment, level, fold, game."""
    rows = []
    for m in models_ or models(cfg):
        for e in experiments:
            for level, dirs in sources(cfg, m, e):
                for f, d in dirs.items():
                    rows.append({"model": m, "experiment": e, "level": level, "fold": f,
                                 "game": load_game(d / "vdict.pkl", v_empty)})
    return pd.DataFrame(rows)


def load_ps(cfg, model: str, experiment: str = "main") -> pd.DataFrame:
    """Per-fold Perceptual Score: model, experiment, level, fold, modality, PS_raw."""
    rows = []
    for level, dirs in sources(cfg, model, experiment):
        for f, d in dirs.items():
            for r in pd.read_csv(d / "ps.csv").itertuples(index=False):
                rows.append({"model": model, "experiment": experiment, "level": level, "fold": f,
                             "modality": r.modality, "PS_raw": float(r.PS_raw)})
    return pd.DataFrame(rows)


# ── pure game theory ─────────────────────────────────────────────────────────
def harsanyi(v: dict) -> dict:
    out = {}
    for T in _coalitions():
        if T:
            out[T] = sum((-1) ** (len(T) - r) * v[frozenset(S)]
                         for r in range(len(T) + 1) for S in itertools.combinations(sorted(T), r))
    return out


def interaction_split(v: dict, views=VIEW_ORDER) -> dict:
    terms = shapley_interaction_terms(list(views), v)
    out = {}
    for pair in PAIRS:
        t = terms.get(pair) or terms[pair[::-1]]
        total = float(sum(t.values()))
        out[pair] = (total, float(t[0]), total - float(t[0]))
    return out


def phi(v: dict, views=VIEW_ORDER) -> dict:
    return shapley_values(list(views), v)


def identity_errors(v: dict) -> dict:
    """Max |error| of I_ij = Σ_{T⊇{i,j}} d(T)/(|T|-1) and φ_i = Σ_{T∋i} d(T)/|T|."""
    d = harsanyi(v)
    I = interaction_split(v)
    p = phi(v)
    e_I = max(abs(I[pr][0] - sum(x / (len(T) - 1) for T, x in d.items() if set(pr) <= T)) for pr in PAIRS)
    e_phi = max(abs(p[i] - sum(x / len(T) for T, x in d.items() if i in T)) for i in VIEW_ORDER)
    return {"I": e_I, "phi": e_phi}


def phi_frame(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in games.itertuples(index=False):
        for mod, val in phi(r.game).items():
            rows.append({"model": r.model, "experiment": r.experiment, "level": r.level,
                         "fold": r.fold, "modality": mod, "phi": val})
    return pd.DataFrame(rows)


def decomposition(v: dict, views=VIEW_ORDER) -> dict:
    full = frozenset(views)
    p = phi(v, views)
    out = {}
    for m in views:
        mg = v[full] - v[full - {m}]
        out[m] = {"phi": p[m], "phi_grand": mg / len(views), "phi_small": p[m] - mg / len(views),
                  "marginal_grand": mg}
    return out


def decomposition_frame(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in games.itertuples(index=False):
        for mod, d in decomposition(r.game).items():
            rows.append({"model": r.model, "experiment": r.experiment, "level": r.level,
                         "fold": r.fold, "modality": mod, **d})
    return pd.DataFrame(rows)
