"""
The pipeline configuration: configs/paper.yaml (+ the ``smoke`` overlay), resolved paths and
the location of every artefact. Nothing else in the package hard-codes a path.

Artefacts of one job = (model, experiment, level, fold):

    <runs_dir>/<model>/<experiment>[/<level>]/fold<k>/
        model.pt      state_dict of the trained model
        metrics.json  full-coalition test metrics right after training, epochs, timing
        vdict.pkl     the 16-coalition game (per-coalition predictions; see shapley/core.py)
        ps.csv        permutation Perceptual Score               (experiments with ps: true)
        local.npz     per-sample v(S) of the 16 coalitions        (experiments with local: true)

with <level> = sigma_<s> (noise) or k_<kk> (cloud_gap), no level for ``main``.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper.yaml"
EXPERIMENTS = ("main", "noise", "cloud_gap")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load(path=None, smoke: bool = False, runs_dir=None, out_dir=None) -> dict:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    smoke_block = cfg.pop("smoke", {}) or {}
    if smoke:
        cfg = _merge(cfg, smoke_block)
    cfg["smoke"] = bool(smoke)
    cfg["root"] = ROOT
    cfg["data_path"] = ROOT / cfg["data"]["dir"]
    cfg["runs_path"] = ROOT / (runs_dir or cfg["runs_dir"])
    cfg["out_path"] = ROOT / (out_dir or cfg["out_dir"])
    return cfg


def model_config(cfg: dict, model: str) -> dict:
    """The architecture / training recipe of ``model`` (configs/models/<model>.yaml)."""
    mc = yaml.safe_load((ROOT / cfg["models"][model]["config"]).read_text())
    if cfg.get("max_epochs"):
        mc["training"]["max_epochs"] = int(cfg["max_epochs"])
    return mc


def levels(cfg: dict, experiment: str) -> list:
    """Levels of an experiment, in sweep order ([None] for ``main``)."""
    return list(cfg["experiments"][experiment].get("levels") or [None])


def level_name(experiment: str, level) -> str | None:
    if experiment == "noise":
        return f"sigma_{float(level):g}"
    if experiment == "cloud_gap":
        return f"k_{int(level):02d}"
    return None


def level_value(cfg: dict, experiment: str, level) -> float:
    """The x value of a level in tables and figures: sigma, or the share of S2 dates lost."""
    if experiment == "noise":
        return float(level)
    if experiment == "cloud_gap":
        return int(level) / cfg["experiments"]["cloud_gap"]["seq_len"]
    return 0.0


def job_dir(cfg: dict, model: str, experiment: str, level, fold: int) -> Path:
    d = cfg["runs_path"] / model / experiment
    name = level_name(experiment, level)
    if name:
        d = d / name
    return d / f"fold{fold}"


def splits_file(cfg: dict) -> Path:
    return cfg["runs_path"] / "splits.npz"


def out(cfg: dict, kind: str) -> Path:
    """Output folder of the report: figures | tables | data | numbers | local_analysis."""
    p = cfg["out_path"] / kind
    p.mkdir(parents=True, exist_ok=True)
    return p
