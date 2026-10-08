"""Training of one model on one fold (PyTorch Lightning), as in the paper's runs:
Adam, early stopping on the validation CE of the full coalition (min_delta 0.01, patience 5),
the best epoch restored. As in the paper, the validation set is the test fold."""
from __future__ import annotations

import logging
import shutil
import time
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from shapfusion.data.dataset import create_dataloader

for _name in ("pytorch_lightning", "lightning_fabric"):
    logging.getLogger(_name).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*The number of training batches.*")


class _EpochLog(Callback):
    def on_validation_epoch_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        tr, va = m.get("train_objective"), m.get("val_loss_full")
        print(f"    epoch {trainer.current_epoch:3d}  train_objective "
              f"{float(tr) if tr is not None else float('nan'):.4f}  val_loss_full {float(va):.4f}",
              flush=True)


def fit(model, train_ds, val_ds, model_cfg: dict, work_dir: Path, device: str, seed: int) -> dict:
    """Train ``model`` in place; returns {"epochs", "best_val_loss", "train_time_s"}."""
    training = model_cfg["training"]
    pl.seed_everything(seed, workers=False)
    bs, pp = training["batch_size"], training.get("parallel_processes", 2)
    train_dl = create_dataloader(train_ds, batch_size=bs, train=True, parallel_processes=pp)
    val_dl = create_dataloader(val_ds, batch_size=bs, train=False, parallel_processes=pp)
    es_args = training["early_stop_args"]
    early = EarlyStopping(monitor="val_loss_full", **es_args)
    ckpt_dir = Path(work_dir) / "_checkpoint"
    best = ModelCheckpoint(monitor="val_loss_full", mode=es_args["mode"], save_top_k=1,
                           dirpath=ckpt_dir, filename="best")
    gpu = device.startswith("cuda") and torch.cuda.is_available()
    trainer = pl.Trainer(max_epochs=training["max_epochs"], accelerator="gpu" if gpu else "cpu", devices=1,
                         callbacks=[early, best, _EpochLog()], logger=False, enable_progress_bar=False,
                         enable_model_summary=False, default_root_dir=str(work_dir))
    t0 = time.time()
    trainer.fit(model, train_dl, val_dataloaders=val_dl)
    state = torch.load(best.best_model_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state)
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return {"epochs": int(early.stopped_epoch), "best_val_loss": float(early.best_score),
            "train_time_s": round(time.time() - t0, 1)}
