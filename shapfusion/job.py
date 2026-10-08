"""
One job = (model, experiment, level, fold), four stages, each skipped when its output exists:

    train   model.pt, metrics.json   train on the 4 other folds (perturbed for noise / cloud_gap)
    vdict   vdict.pkl                the 16 coalitions on the test fold (shapley/core.py)
    ps      ps.csv                   permutation Perceptual Score     (experiments with ps: true)
    local   local.npz                per-sample v(S) of the 16 coalitions (experiments with local: true)

local.npz (coalition axis first, row m = bitmask, bit k <-> views[k]; row 0 = ∅ is NaN):
    p1 (16, N)       P(class 1 | x_S)  (EmbraceNet: mean over the R draws of the v-dict)
    logit (16, N)    log P(class 1 | x_S) - log P(class 0 | x_S)
    logit64 (16, N)  raw logit difference from a float64 forward pass (models with logit64)
    y_true, sample_idx, identifier, lon, lat  (N,)   in the order of the test fold
"""
from __future__ import annotations

import gc
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from shapfusion import config as C
from shapfusion.data import perturb
from shapfusion.data.dataset import create_dataloader, load_dataset
from shapfusion.data.splits import get_splits
from shapfusion.models.build import build_model, class_weights
from shapfusion.shapley.characteristic import compute_metrics
from shapfusion.shapley.core import members, run_subset_inference
from shapfusion.shapley.perceptual_score import perceptual_score

STAGES = ("train", "vdict", "ps", "local")
OUTPUT = {"train": "model.pt", "vdict": "vdict.pkl", "ps": "ps.csv", "local": "local.npz"}


@dataclass(frozen=True)
class Job:
    model: str
    experiment: str
    level: object
    fold: int

    def __str__(self):
        lv = C.level_name(self.experiment, self.level)
        return f"{self.model}/{self.experiment}{'/' + lv if lv else ''}/fold{self.fold}"


def all_jobs(cfg, models=None, experiments=None, levels=None, folds=None) -> list:
    """Every job in run order: main (all models) -> noise -> cloud_gap; model, level, fold."""
    out = []
    for exp in C.EXPERIMENTS:
        if experiments and exp not in experiments:
            continue
        for model in cfg["experiments"][exp]["models"]:
            if models and model not in models:
                continue
            for lv in C.levels(cfg, exp):
                if levels and lv is not None and not any(float(lv) == float(x) for x in levels):
                    continue
                for fold in cfg["folds"]:
                    if folds is None or fold in folds:
                        out.append(Job(model, exp, lv, fold))
    return out


def stages_of(cfg, job: Job) -> list:
    spec = cfg["experiments"][job.experiment]
    return [s for s in STAGES if s in ("train", "vdict") or spec.get(s, False)]


def done(cfg, job: Job, stage: str) -> bool:
    return (C.job_dir(cfg, job.model, job.experiment, job.level, job.fold) / OUTPUT[stage]).exists()


def _atomic(path: Path, write):
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)


def _savez(path, arrays):
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **arrays)


# ── inference helpers ────────────────────────────────────────────────────────
def test_tensors(data_te, batch_size):
    """The whole (perturbed) test fold as {view: tensor}, in test order."""
    chunks = {}
    for batch in create_dataloader(data_te, batch_size=batch_size, train=False):
        for k, v in batch["views"].items():
            chunks.setdefault(k, []).append(v)
    return {k: torch.cat(v) for k, v in chunks.items()}


def make_predict(model, X, batch_size, device, dtype=torch.float32):
    """predict(subset, out_norm) -> (N, C) numpy, same batches as a DataLoader pass."""
    model.to(device=device, dtype=dtype)
    n = next(iter(X.values())).shape[0]

    def predict(subset, out_norm="softmax"):
        model.eval()
        outs = []
        with torch.no_grad():
            for a in range(0, n, batch_size):
                views = {k: v[a:a + batch_size].to(device=device, dtype=dtype) for k, v in X.items()}
                out = model(views, out_norm=out_norm, inference_views=list(subset),
                            missing_method=model.missing_method)["prediction"]
                outs.append(out.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)
    return predict


# ── the runner ───────────────────────────────────────────────────────────────
class Runner:
    def __init__(self, cfg, device: str | None = None, log=print):
        self.cfg, self.log = cfg, log
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.views = cfg["data"]["views"]
        self.data = load_dataset(cfg["data_path"], cfg["data"]["name"], self.views, cfg.get("n_samples"))
        self.splits = get_splits(cfg, self.data)
        self.model_cfgs = {}

    def model_cfg(self, model):
        if model not in self.model_cfgs:
            self.model_cfgs[model] = C.model_config(self.cfg, model)
        return self.model_cfgs[model]

    def fold_data(self, job: Job):
        pert = perturb.for_job(self.cfg, job.experiment, job.level, job.fold)
        test_ids = self.splits[job.fold]
        tr = self.data.fold(test_ids, train=True, perturbation=pert)
        te = self.data.fold(test_ids, train=False, perturbation=pert)
        return tr, te

    def load_model(self, job, tr, path):
        model = build_model(tr, self.model_cfg(job.model), class_weights(tr.get_all_labels()))
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.set_missing_info(None, **self.model_cfg(job.model)["training"]["missing_method"])
        return model.eval()

    def run(self, job: Job, stages=STAGES, force=False):
        cfg = self.cfg
        d = C.job_dir(cfg, job.model, job.experiment, job.level, job.fold)
        todo = [s for s in stages_of(cfg, job) if s in stages and (force or not done(cfg, job, s))]
        if not todo:
            return False
        if any(s != "train" for s in todo) and "train" not in todo and not done(cfg, job, "train"):
            raise FileNotFoundError(f"{job}: no trained model ({d / 'model.pt'}); run the train stage first")
        d.mkdir(parents=True, exist_ok=True)
        self.log(f"[{time.strftime('%H:%M:%S')}] {job}: {', '.join(todo)}")
        tr, te = self.fold_data(job)
        mcfg = self.model_cfg(job.model)
        bs = mcfg["training"]["batch_size"]
        timings = {}

        if "train" in todo:
            from shapfusion.train import fit
            t0 = time.time()
            model = build_model(tr, mcfg, class_weights(tr.get_all_labels()))
            info = fit(model, tr, te, mcfg, d, self.device, seed=cfg["train_seed"] + job.fold)
            state = model.state_dict()
            _atomic(d / "model.pt", lambda p: torch.save(state, p))
            model.set_missing_info(None, **mcfg["training"]["missing_method"])
            out = model.transform(create_dataloader(te, batch_size=bs, train=False), out_norm="softmax",
                                  device=self.device)
            metrics = {**compute_metrics(te.get_all_labels(), out["prediction"]), **info}
            _atomic(d / "metrics.json", lambda p: p.write_text(json.dumps(metrics, indent=2)))
            self.log(f"    f1_weighted {metrics['f1_weighted']:.4f} after {info['epochs']} epochs")
            timings["train"] = round(time.time() - t0, 1)
            del model

        model = self.load_model(job, tr, d / "model.pt") if any(s != "train" for s in todo) else None
        draws = int(cfg["models"][job.model].get("draws", 1))
        spec = cfg["experiments"][job.experiment]

        if "vdict" in todo:
            t0 = time.time()
            # deterministic models: the training batch size, as the paper's runs (bit-identical
            # outputs); stochastic draws do not depend on the batching -> large batches
            ibs = bs if draws == 1 else int(cfg["inference"]["stochastic_batch_size"])
            X = test_tensors(te, ibs)
            predict = make_predict(model, X, ibs, self.device)
            torch.manual_seed(cfg["inference"]["seed"] + job.fold)
            full = None if draws > 1 else predict(self.views, "softmax")
            vd = run_subset_inference(predict, self.views, te.get_all_labels(), tr.get_all_labels(),
                                      draws=draws, full_pred=full, store_proba=bool(spec.get("local")))
            _atomic(d / "vdict.pkl", lambda p: p.write_bytes(pickle.dumps(vd)))
            timings["vdict"] = round(time.time() - t0, 1)

        if "ps" in todo:
            t0 = time.time()
            y = te.get_all_labels()
            out = model.transform(create_dataloader(te, batch_size=bs, train=False), out_norm="softmax",
                                  device=self.device)
            f1_all = compute_metrics(y, out["prediction"])[cfg["perceptual_score"]["metric"]]
            rows = perceptual_score(model, te, y, self.views, bs, f1_all,
                                    n_perm=cfg["perceptual_score"]["n_perm"],
                                    seed=spec["ps_seed"] + job.fold,
                                    metric=cfg["perceptual_score"]["metric"], device=self.device)
            df = pd.DataFrame(rows).assign(fold=job.fold)
            _atomic(d / "ps.csv", lambda p: df.to_csv(p, index=False))
            timings["ps"] = round(time.time() - t0, 1)

        if "local" in todo:
            t0 = time.time()
            arrays = self.local_arrays(job, model, te, d)
            _atomic(d / "local.npz", lambda p: _savez(p, arrays))
            timings["local"] = round(time.time() - t0, 1)

        info = json.loads((d / "job.json").read_text()) if (d / "job.json").exists() else {}
        info.update({"job": str(job), "device": self.device, "torch": torch.__version__,
                     "draws": draws, "timings_s": {**info.get("timings_s", {}), **timings}})
        (d / "job.json").write_text(json.dumps(info, indent=2))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def local_arrays(self, job, model, te, d) -> dict:
        """Per-sample games of the test fold, from the v-dict (+ float64 logits)."""
        vd = pickle.loads((d / "vdict.pkl").read_bytes())
        views = self.views
        y = np.asarray(vd["y_true"]).ravel()
        n = len(y)
        p1 = np.full((16, n), np.nan)
        logit = np.full((16, n), np.nan)
        for S, rec in vd["coalitions"].items():
            if not S:
                continue
            m = sum(1 << views.index(v) for v in S)
            if "pred" in rec:
                P = np.asarray(rec["pred"], dtype=np.float64)
                p1[m], logit[m] = P[:, 1], np.log(P[:, 1]) - np.log(P[:, 0])
            else:
                p1[m], logit[m] = rec["p1_mean"], np.log(rec["p1_mean"]) - np.log(rec["p0_mean"])
        lonlat = te.lonlat().astype(np.float64)
        arrays = {"p1": p1, "logit": logit, "y_true": y.astype(np.int64),
                  "sample_idx": np.arange(n, dtype=np.int64),
                  "identifier": np.asarray(te.get_all_identifiers()).astype(str),
                  "lon": lonlat[:, 0], "lat": lonlat[:, 1]}
        if self.cfg["models"][job.model].get("logit64"):
            bs = self.cfg["inference"]["logit64_batch_size"]
            predict = make_predict(model, test_tensors(te, bs), bs, self.device, dtype=torch.float64)
            l64 = np.full((16, n), np.nan)
            for m in range(1, 16):
                z = predict([v for v in views if v in members(m, views)], "")
                l64[m] = z[:, 1] - z[:, 0]
            arrays["logit64"] = l64
            model.to(dtype=torch.float32)
        return arrays
