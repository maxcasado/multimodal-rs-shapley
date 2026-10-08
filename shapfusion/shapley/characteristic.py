"""
Characteristic functions of the GLOBAL (fold-level) game: v(S) = a classification metric of
the model restricted to the sensors S, evaluated on the test fold. The paper uses
f1_weighted; the v-dicts store predictions, so the metric is a post-processing choice.

The empty coalition is not a model. The paper's v(∅) for f1_weighted is the expected
f1_weighted of the uniform (coin-flip) classifier on a binary task with positive share p:
    v(∅) = 2p²/(2p+1) + 2(1-p)²/(3-2p),   p = 0.65744 (69,800 samples)  ->  0.5127
applied when the v-dicts are read (core.scalar_vdict). The v-dicts themselves store, as
∅ record, the constant TRAIN-majority prediction (any metric can be evaluated on it).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)

DEFAULT_METRIC = "f1_weighted"


def _labels(proba: np.ndarray) -> np.ndarray:
    return np.asarray(proba).argmax(axis=-1)


CHAR_FUNCS = {
    "accuracy":          lambda y, p: accuracy_score(y, _labels(p)),
    "f1_macro":          lambda y, p: f1_score(y, _labels(p), average="macro", zero_division=0),
    "f1_weighted":       lambda y, p: f1_score(y, _labels(p), average="weighted", zero_division=0),
    "precision_macro":   lambda y, p: precision_score(y, _labels(p), average="macro", zero_division=0),
    "recall_macro":      lambda y, p: recall_score(y, _labels(p), average="macro", zero_division=0),
    "mcc":               lambda y, p: matthews_corrcoef(y, _labels(p)),
    "balanced_accuracy": lambda y, p: balanced_accuracy_score(y, _labels(p)),
}
METRICS = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]


def char_value(y_true, proba, metric: str) -> float:
    return float(CHAR_FUNCS[metric](y_true, proba))


def compute_metrics(y_true, proba) -> dict:
    return {m: char_value(y_true, proba, m) for m in METRICS}


def train_baseline_record(y_train, y_test) -> dict:
    """∅ record: the TRAIN-majority class predicted for every test sample, one-hot."""
    y_test = np.asarray(y_test)
    maj = int(np.bincount(np.asarray(y_train).ravel().astype(int)).argmax())
    n_classes = int(max(np.asarray(y_train).max(), y_test.max())) + 1
    pred = np.zeros((len(y_test), n_classes), dtype=float)
    pred[:, maj] = 1.0
    return {"pred": pred}


UNIFORM_P = 0.65744


def uniform_v_empty_f1(p: float = UNIFORM_P) -> float:
    """Expected f1_weighted of the uniform binary classifier, positive share p."""
    return 2 * p ** 2 / (2 * p + 1) + 2 * (1 - p) ** 2 / (3 - 2 * p)


V_EMPTY_F1 = uniform_v_empty_f1()          # 0.5127, the paper's v(∅)
