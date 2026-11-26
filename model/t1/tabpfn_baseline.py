"""TabPFN v2 as an extra comparator of the T1 baselines grid (handoff 2026-11-26 pack p4).

Exploratory, our choice, not a PI decision. Opt-in: only a grid config whose ``families`` name ``tabpfn`` reaches this
module (``tools/t1_grid_search.py``); no Table 1 code path imports it. MODERN_BENCHMARK_STRATEGY lists TabPFN
(Hollmann et al. 2025, Nature, doi 10.1038/s41586-024-08328-6) as the tabular foundation-model reference "on tasks
within its supported regime"; that paper's model is TabPFN v2, so v2 is used (later versions are gated behind a
license acceptance on Hugging Face). Declared choices:

* package ``tabpfn==10.2.0`` (pinned in main's pyproject), model version v2, the default v2 checkpoints
  ``tabpfn-v2-classifier.ckpt`` / ``tabpfn-v2-regressor.ckpt`` (Prior-Labs/TabPFN-v2-clf, -reg on Hugging Face) with
  the sha256 below; a trial refuses another file;
* CPU inference, ``n_estimators`` from the grid (one fixed value, no HPO), ``random_state`` = the trial seed,
  ``ignore_pretraining_limits`` true (the package refuses > 1000 rows on CPU otherwise), one preprocessing job and the
  torch threads of the grid study;
* regime: at most ``max_train_rows`` (3000) training rows, ``max_features`` (500) features and 10 classes; a trial
  outside it fails before fitting (the job lists contain only tasks inside it);
* inputs are the grid's own preprocessed splits (train-fitted imputation, scaling and one-hot encoding), so the model
  sees exactly the tensors the other families see; the validation stream is only scored, never used for fitting;
* nothing is trained, so a trial unit keeps no weights: its ``model`` state names the pinned checkpoint, its sha256,
  the package version and the estimator settings; the context rows are the recorded ``data`` state.

Weights: ``<NN-KNN root>/models/tabpfn/`` (the parent of ``datasets.paths.datasets_root()``), or ``NNKNN_TABPFN_DIR``.
``python tools/t1_tabpfn_weights.py`` downloads and checks them once per host.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

PACKAGE_VERSION = "10.2.0"
MODEL_VERSION = "v2"
CHECKPOINTS = {
    "classification": ("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier.ckpt",
                       "f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f"),
    "regression": ("Prior-Labs/TabPFN-v2-reg", "tabpfn-v2-regressor.ckpt",
                   "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"),
}
MAX_CLASSES = 10
PARAMETERS = frozenset({"model_version", "n_estimators", "max_train_rows", "max_features"})


def weights_dir() -> Path:
    env = os.environ.get("NNKNN_TABPFN_DIR")
    if env:
        return Path(env)
    from datasets.paths import datasets_root
    return datasets_root().parent / "models" / "tabpfn"


@lru_cache(maxsize=None)
def _sha256(path: str, size: int, mtime: float) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint(task: str) -> tuple[Path, str]:
    """Path and sha256 of the pinned checkpoint of ``task``; refuses a missing or different file."""
    _, name, expected = CHECKPOINTS[task]
    path = weights_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"TabPFN checkpoint {path} is missing; run tools/t1_tabpfn_weights.py on this host")
    st = path.stat()
    digest = _sha256(str(path), st.st_size, st.st_mtime)
    if digest != expected:
        raise ValueError(f"TabPFN checkpoint {path} has sha256 {digest}, expected the pinned {expected}")
    return path, digest


def check_version() -> str:
    version = importlib.metadata.version("tabpfn")
    if version != PACKAGE_VERSION:
        raise RuntimeError(f"tabpfn {version} is installed; this comparator is pinned to {PACKAGE_VERSION}")
    return version


def check_parameters(params: dict) -> None:
    unknown = set(params) - PARAMETERS
    if unknown:
        raise ValueError(f"unknown tabpfn parameters {sorted(unknown)}; expected {sorted(PARAMETERS)}")
    if params.get("model_version", MODEL_VERSION) != MODEL_VERSION:
        raise ValueError(f"only TabPFN {MODEL_VERSION} is supported (later versions are gated)")
    n = params.get("n_estimators")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("tabpfn n_estimators must be a positive integer")


def in_regime(n_train: int, n_features: int, num_classes: int | None, params: dict) -> str | None:
    """None when a task is inside the declared regime, else the reason it is not."""
    if n_train > int(params["max_train_rows"]):
        return f"{n_train} training rows > {params['max_train_rows']}"
    if n_features > int(params["max_features"]):
        return f"{n_features} features > {params['max_features']}"
    if num_classes is not None and num_classes > MAX_CLASSES:
        return f"{num_classes} classes > {MAX_CLASSES}"
    return None


def fit(params: dict, data: dict, seed: int):
    """A fitted TabPFN v2 estimator on ``data`` (the grid's train/validation dict) and its info."""
    check_parameters(params)
    reason = in_regime(len(data["y_train"]), int(data["X_train"].shape[1]),
                       data["classes"] if data["task"] == "classification" else None, params)
    if reason is not None:
        raise ValueError(f"task outside the declared TabPFN regime: {reason}")
    version = check_version()
    path, digest = checkpoint(data["task"])
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
    ctor = TabPFNClassifier if data["task"] == "classification" else TabPFNRegressor
    model = ctor.create_default_for_version(
        ModelVersion.V2, model_path=str(path), n_estimators=int(params["n_estimators"]), device="cpu",
        ignore_pretraining_limits=True, random_state=int(seed), n_preprocessing_jobs=1, show_progress_bar=False)
    X = data["X_train"].numpy().astype(np.float32)
    y = data["y_train"].numpy()
    model.fit(X, y)
    info = dict(tabpfn_version=version, model_version=MODEL_VERSION, checkpoint=path.name, checkpoint_sha256=digest,
                n_estimators=int(params["n_estimators"]), device="cpu", ignore_pretraining_limits=True,
                context_rows=int(len(y)))
    return model, info


def predict(model, X, task: str, num_classes: int | None = None):
    """Class probabilities over all ``num_classes`` labels (zero for a class absent from the context) or the
    regression mean."""
    Xn = X.numpy().astype(np.float32)
    if task != "classification":
        return np.asarray(model.predict(Xn), dtype=np.float64).reshape(-1)
    return expand_proba(model.predict_proba(Xn), model.classes_, num_classes)


def expand_proba(proba, classes, num_classes):
    """``proba`` over ``classes`` (the labels seen in training) as columns 0..num_classes-1."""
    proba = np.asarray(proba, dtype=np.float64)
    classes = np.asarray(classes).astype(np.int64)
    if num_classes is None or (len(classes) == num_classes and np.array_equal(classes, np.arange(num_classes))):
        return proba
    out = np.zeros((proba.shape[0], int(num_classes)))
    out[:, classes] = proba
    return out


def state(info: dict) -> dict:
    """The recorded ``model`` state of a TabPFN trial: no weights (nothing is trained), the pinned checkpoint."""
    return {"format": "tabpfn_reference", **info}
