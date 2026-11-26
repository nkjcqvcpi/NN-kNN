"""Data streams for T1 (contract section 11).

Four disjoint streams: training (cases + retrieval training), optional
maintenance/audit, validation (early stopping / model selection), and an
untouched test set. Preprocessing (feature scaler, regression target scaling,
regression cohort bins) is fit on the training stream only.

Synthetic diagnostic tasks with *known* structure are provided for Phase 1
(plan section 13): label corruption, redundancy (near-duplicates), a rare
sub-cluster and a shifted sub-domain, each with ground-truth case flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import hashlib
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datasets.paths import datasets_root

CLASSIFICATION_SETS = ("iris", "wine", "breast_cancer", "balance", "digits", "zebra", "zebra_special")
REGRESSION_SETS = (
    "airfoil",
    "car",
    "yacht",
    "energy_efficiency",
    "student_performance",
    "abalone",
    "body_fat",
    "diabets",
    "califonia_housing",
    "bike_sharing",
    "wine",
)


@dataclass
class T1Data:
    name: str
    task_type: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    X_maint: torch.Tensor | None = None
    y_maint: torch.Tensor | None = None
    num_classes: int | None = None
    reg_bins: np.ndarray | None = None
    y_scale: tuple[float, float] | None = None  # regression (mean, std) fit on train
    case_truth: dict[str, np.ndarray] = field(default_factory=dict)  # synthetic ground-truth flags per train row
    test_groups: dict[str, np.ndarray] = field(default_factory=dict)  # e.g. shifted-subdomain test mask
    meta: dict[str, Any] = field(default_factory=dict)
    nominal_train: dict[str, list[Any]] | None = None
    nominal_val: dict[str, list[Any]] | None = None
    nominal_test: dict[str, list[Any]] | None = None
    nominal_maint: dict[str, list[Any]] | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type,
            "n_train": int(self.X_train.shape[0]),
            "n_maint": 0 if self.X_maint is None else int(self.X_maint.shape[0]),
            "n_val": int(self.X_val.shape[0]),
            "n_test": int(self.X_test.shape[0]),
            "n_features": int(self.X_train.shape[1]),
            "num_classes": self.num_classes,
            "reg_bins": None if self.reg_bins is None else self.reg_bins.tolist(),
            **{k: v for k, v in self.meta.items() if k != "true_labels"},
        }


def _load_raw(name: str, task_type: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if name.startswith(("openml_", "tabarena_")):
        import openml
        task = openml.tasks.get_task(int(name.split("_")[1]), download_splits=True) if name.startswith("tabarena_") else None
        ds = openml.datasets.get_dataset(task.dataset_id if task else int(name.split("_")[1]), download_data=True)
        frame, target, _, _ = ds.get_data(target=task.target_name if task else ds.default_target_attribute, dataset_format="dataframe")
        if task_type == "classification":
            target = pd.Categorical(target).codes.astype(np.int64)
        source_meta = {"source": "openml", "openml_id": ds.dataset_id,
            "openml_version": ds.version, "name_on_source": ds.name,
            "license": ds.licence, "protocol": "T1 reconstructed holdout, not official TabArena splits"}
        if task:
            repeats, folds, _ = task.get_split_dimensions()
            repeat, fold = (seed // folds) % repeats, seed % folds
            train_idx, test_idx = task.get_train_test_split_indices(repeat=repeat, fold=fold)
            source_meta.update(openml_task_id=task.task_id, outer_repeat=repeat, outer_fold=fold,
                outer_train_indices=train_idx.tolist(), outer_test_indices=test_idx.tolist(),
                protocol="OpenML official outer split; train-only inner holdout; no TabArena bagging/leaderboard equivalence")
        return frame, np.asarray(target).reshape(-1), source_meta
    if task_type == "regression" and name == "student_performance":
        frame = pd.read_csv(datasets_root() / "student-por.csv", sep=";")
        return frame.drop(columns="G3"), frame.G3.to_numpy(), {"source": "student-por.csv", "preprocessing": "training_only_categories"}
    if task_type == "regression" and name == "body_fat":
        frame = pd.read_csv(datasets_root() / "bodyfat.csv")
        x = frame.drop(columns=["BodyFat", "Density"]).copy()
        x["Bmi"] = 703 * x.Weight / x.Height.pow(2)
        x["ACratio"] = x.Abdomen / x.Chest
        x["HTratio"] = x.Hip / x.Thigh
        x = x.drop(columns=["Weight", "Height", "Abdomen", "Chest", "Hip", "Thigh"])
        return x, frame.Density.to_numpy(), {"source": "bodyfat.csv", "target": "Density (legacy target)",
            "outlier_policy": "no whole-dataset statistical filtering"}
    if task_type == "regression" and name == "diabets":
        from sklearn.datasets import load_diabetes
        ds = load_diabetes(scaled=False)
        return ds.data, ds.target, {"source": "sklearn.load_diabetes(scaled=False)"}
    if task_type == "classification":
        from datasets.classification_data import load_small_classification_dataset

        st = load_small_classification_dataset(name, seed=seed)
        return st["X"].numpy().astype(np.float32), st["y"].numpy().astype(np.int64), {"source": "datasets.classification_data"}
    from datasets.reg_data import Reg_data
    # Public remote legacy data are cached before any learned preprocessing.
    # Repeated HPO seeds/families must not refetch UCI at every fit.
    cache = datasets_root() / "t1_cache" / f"{name}.npz"
    cacheable = name in {"abalone", "califonia_housing", "bike_sharing", "wine"}
    if cacheable and not cache.exists():
        # The cache used to live in the worktree (<repo>/datasets/t1_cache). If it
        # was not moved, rebuilding from the network would silently record a new
        # raw_snapshot_sha256, so stop and ask for the move instead.
        from pathlib import Path
        legacy = Path(__file__).resolve().parents[2] / "datasets" / "t1_cache" / f"{name}.npz"
        if legacy.exists():
            raise FileNotFoundError(
                f"T1 raw-snapshot cache {cache} is missing but the old copy {legacy} exists. "
                f"Move {legacy.parent} into {cache.parent} byte for byte (do not regenerate it), then rerun."
            )
    if cacheable and cache.exists():
        with np.load(cache, allow_pickle=False) as raw:
            X, y = raw["X"], raw["y"]
    else:
        X, y = Reg_data(name)
        if cacheable:
            cache.parent.mkdir(parents=True, exist_ok=True)
            # Avoid colliding writers when separate dataset workers start.
            import os
            temporary = cache.with_name(f"{name}.{os.getpid()}.npz")
            np.savez_compressed(temporary, X=np.asarray(X, dtype=np.float32), y=np.asarray(y, dtype=np.float32))
            temporary.replace(cache)
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32).reshape(-1),
        {"source": "datasets.reg_data", **({"raw_snapshot_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()} if cacheable else {})},
    )


def make_splits(
    name: str,
    task_type: str,
    seed: int,
    *,
    test_frac: float,
    val_frac: float,
    maint_frac: float = 0.0,
    reg_cohort_bins: int = 5,
    max_train: int | None = None,
    synthetic: dict[str, Any] | None = None,
    split_mode: str = "random",
    shift_quantile: float = 0.7,
) -> T1Data:
    """Train/validation/test(/maintenance) streams of one dataset and seed.

    ``split_mode`` "random" (default) is the recorded split. "covariate_pc1" (opt-in, handoff 2026-11-26 pack p4,
    exploratory, our choice) is a covariate-shift split: see ``covariate_pc1_order``. The training side is the lower
    ``shift_quantile`` of the first principal component of the standardized numeric features over all rows, the test
    stream is the rest; validation (and maintenance, ``max_train``) are then carved from the training side exactly as
    in the random split (validation = ``val_frac / (1 - test_frac)`` of the training side, seeded). OpenML outer
    splits are not used in this mode; synthetic generators have fixed designs and refuse it.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"unknown split_mode {split_mode!r}; expected one of {sorted(SPLIT_MODES)}")
    if split_mode != "random":
        if name.startswith("synthetic"):
            raise ValueError("synthetic generators have fixed designs; split_mode must be 'random'")
        if isinstance(shift_quantile, bool) or not 0.0 < float(shift_quantile) < 1.0:
            raise ValueError("shift_quantile must lie strictly between 0 and 1")
    if name.startswith("synthetic"):
        if task_type != "classification":
            raise ValueError("synthetic diagnostics support classification only")
        if name == "synthetic_nominal":
            if synthetic:
                raise ValueError("nominal engineering data has its own fixed declared generator")
            return make_nominal_synthetic(seed)
        return make_synthetic(name, seed, **(synthetic or {}))
    if synthetic is not None:
        raise ValueError("synthetic parameters cannot be applied to real datasets")
    X, y, meta = _load_raw(name, task_type, seed)
    idx = np.arange(len(y))
    strat = y if task_type == "classification" else None
    shift_groups = None
    if split_mode == "covariate_pc1":
        meta.pop("outer_train_indices", None)
        meta.pop("outer_test_indices", None)
        for key in ("outer_repeat", "outer_fold"):
            meta.pop(key, None)
        order, info = covariate_pc1_order(X)
        n_low = int(round(float(shift_quantile) * len(order)))
        if not 0 < n_low < len(order):
            raise ValueError("shift_quantile leaves an empty training or test side")
        tr_idx, te_idx = np.sort(order[:n_low]), np.sort(order[n_low:])
        # test rows in PC1 order: the half nearer the training side and the far half
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        far = rank[te_idx] >= n_low + (len(te_idx) + 1) // 2
        shift_groups = {"shift_near": ~far, "shift_far": far}
        meta["protocol"] = ("T1 covariate-shift split (handoff 2026-11-26 p4): training side = lower shift_quantile of "
                            "PC1 of the standardized numeric features over all rows; test = the rest; OpenML outer splits "
                            "not used")
        meta["shift"] = {"split_mode": split_mode, "shift_quantile": float(shift_quantile),
                         "threshold": float(info["scores"][order[n_low - 1]]), **{k: v for k, v in info.items() if k != "scores"}}
    elif "outer_train_indices" in meta:
        tr_idx, te_idx = np.asarray(meta.pop("outer_train_indices")), np.asarray(meta.pop("outer_test_indices"))
    else:
        tr_idx, te_idx = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=strat)
    strat_tr = y[tr_idx] if task_type == "classification" else None
    rel_val = val_frac / (1.0 - test_frac)
    if split_mode == "covariate_pc1" and strat_tr is not None and not _stratifiable(strat_tr, rel_val):
        strat_tr = None  # a class with < 2 training-side rows (or too few validation rows): unstratified, recorded
    if split_mode == "covariate_pc1":
        meta["shift"]["validation_stratified"] = strat_tr is not None
    tr_idx, va_idx = train_test_split(tr_idx, test_size=rel_val, random_state=seed + 1, stratify=strat_tr)
    mt_idx = None
    if maint_frac > 0:
        strat_tr = y[tr_idx] if task_type == "classification" else None
        rel_m = maint_frac / (1.0 - test_frac - val_frac)
        if split_mode == "covariate_pc1" and strat_tr is not None and not _stratifiable(strat_tr, rel_m):
            strat_tr = None
        tr_idx, mt_idx = train_test_split(tr_idx, test_size=rel_m, random_state=seed + 2, stratify=strat_tr)
    if max_train is not None and len(tr_idx) > max_train:
        strat_tr = y[tr_idx] if task_type == "classification" else None
        if (split_mode == "covariate_pc1" and strat_tr is not None
                and not _stratifiable(strat_tr, 1.0 - max_train / len(tr_idx))):
            strat_tr = None
        tr_idx, _ = train_test_split(tr_idx, train_size=max_train, random_state=seed + 3, stratify=strat_tr)
    if isinstance(X, pd.DataFrame):
        numeric = X.select_dtypes(include="number").columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        # Explicitly retain nominal differences in the fitted representation.
        # Category vocabulary and missing-value estimates never use val/test.
        scaler = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler()), numeric),
            ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical)], sparse_threshold=0.)
        scaler.fit(X.iloc[tr_idx])
        t = lambda a: torch.tensor(scaler.transform(X.iloc[a]), dtype=torch.float32)
        meta["nominal_coverage"] = {c: "joint_one_hot_representation_train_fitted" for c in categorical}
    else:
        scaler = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler()).fit(X[tr_idx])
        t = lambda a: torch.tensor(scaler.transform(X[a]), dtype=torch.float32)
    meta["split_indices"] = {"train": tr_idx.tolist(), "validation": va_idx.tolist(), "test": te_idx.tolist(),
        "maintenance": [] if mt_idx is None else mt_idx.tolist()}
    meta["preprocessing_fit_stream"] = "train_only"
    if task_type == "classification":
        yt = lambda a: torch.tensor(y[a], dtype=torch.long)  # noqa: E731
        num_classes = int(len(np.unique(y)))
        bins = None
        y_scale = None
    else:
        mu, sd = float(y[tr_idx].mean()), float(y[tr_idx].std() + 1e-8)
        yt = lambda a: torch.tensor((y[a] - mu) / sd, dtype=torch.float32)  # noqa: E731
        num_classes = None
        qs = np.linspace(0, 1, reg_cohort_bins + 1)[1:-1]
        bins = np.quantile((y[tr_idx] - mu) / sd, qs)
        y_scale = (mu, sd)
    data = T1Data(
        name=name,
        task_type=task_type,
        X_train=t(tr_idx),
        y_train=yt(tr_idx),
        X_val=t(va_idx),
        y_val=yt(va_idx),
        X_test=t(te_idx),
        y_test=yt(te_idx),
        X_maint=None if mt_idx is None else t(mt_idx),
        y_maint=None if mt_idx is None else yt(mt_idx),
        num_classes=num_classes,
        reg_bins=bins,
        y_scale=y_scale,
        meta={**meta, "split_seed": seed, "test_frac": test_frac, "val_frac": val_frac, "maint_frac": maint_frac},
    )
    if shift_groups is not None:
        data.test_groups = {k: np.asarray(v, dtype=bool) for k, v in shift_groups.items()}
        if task_type == "classification":
            data.meta["shift"]["classes_missing_from_training"] = sorted(
                int(c) for c in set(np.unique(y).tolist()) - set(np.unique(y[tr_idx]).tolist()))
    return data


SPLIT_MODES = frozenset({"random", "covariate_pc1"})


def _stratifiable(labels, test_size) -> bool:
    """Whether sklearn can stratify a split of ``labels`` with this test share: every class has >= 2 rows and both
    sides can hold one row of every class."""
    _, counts = np.unique(labels, return_counts=True)
    n, k = len(labels), len(counts)
    n_test = int(np.ceil(test_size * n))
    return bool(counts.min() >= 2 and n_test >= k and n - n_test >= k)


def covariate_pc1_order(X) -> tuple[np.ndarray, dict[str, Any]]:
    """Rows ordered by their score on the first principal component of the standardized numeric features.

    Numeric features only (pandas number dtypes; every column of an array); missing values take the column median;
    constant columns are dropped; each column is standardized with its mean and population std over all rows. PC1 is
    the first right singular vector of that matrix, its sign fixed so that the loading of largest magnitude is positive
    (first such column on ties). Rows are ordered by score, ties by row index. Returns the order and a description
    (``scores``, ``features``, ``loadings``, ``explained_variance_ratio``).
    """
    if isinstance(X, pd.DataFrame):
        cols = X.select_dtypes(include="number").columns.tolist()
        M = X[cols].to_numpy(dtype=np.float64, na_value=np.nan)
        names = [str(c) for c in cols]
    else:
        M = np.asarray(X, dtype=np.float64)
        names = [str(i) for i in range(M.shape[1])]
    if M.ndim != 2 or M.shape[1] == 0:
        raise ValueError("covariate_pc1 needs at least one numeric feature")
    med = np.nanmedian(np.where(np.isnan(M).all(0, keepdims=True), 0.0, M), axis=0)
    M = np.where(np.isnan(M), med[None, :], M)
    sd = M.std(0)
    keep = sd > 1e-12
    if not keep.any():
        raise ValueError("covariate_pc1 needs a non-constant numeric feature")
    Z = (M[:, keep] - M[:, keep].mean(0)) / sd[keep]
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    v = Vt[0]
    j = int(np.argmax(np.abs(v)))
    if v[j] < 0:
        v = -v
    scores = Z @ v
    order = np.lexsort((np.arange(len(scores)), scores))
    var = S ** 2
    return order, {"scores": scores, "features": [n for n, k in zip(names, keep) if k],
                   "loadings": [float(a) for a in v], "explained_variance_ratio": float(var[0] / var.sum())}


def make_synthetic(name: str, seed: int, *, generator_version="legacy",
                   center_scale=3.0, within_std=1.0, corruption_rate=0.15,
                   duplicate_count=60, rare_count=12, shift=1.5,
                   boundary_margin=1.0) -> T1Data:
    """``synthetic_diag``: 3-class Gaussian task with known corruption, redundancy, rare and shift.

    Ground truth per training row:
      corrupted  - label flipped to another class (15% of the base rows)
      duplicate  - near-copy of another base row (redundant)
      rare       - member of a small, far sub-cluster of class 2
    Test rows carry ``shifted`` for a sub-domain whose inputs are displaced.
    Relevant features: first 4 of 8 (the rest are noise).
    """
    if generator_version not in {"legacy", "independent_v1"}:
        raise ValueError("unknown synthetic generator_version")
    for key, value in {"center_scale": center_scale, "within_std": within_std,
                       "corruption_rate": corruption_rate, "shift": shift,
                       "boundary_margin": boundary_margin}.items():
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if center_scale <= 0 or within_std <= 0 or not 0 <= corruption_rate <= 1 or boundary_margin < 0:
        raise ValueError("invalid synthetic scale, corruption rate or boundary margin")
    for key, value in {"duplicate_count": duplicate_count, "rare_count": rare_count}.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    n_base = 360
    clean_count = n_base - int(corruption_rate * n_base)
    available = clean_count if generator_version == "legacy" else n_base
    if duplicate_count > available:
        raise ValueError("duplicate_count exceeds distinct available source rows")
    rng = np.random.default_rng(seed)
    # Separate streams make noise/duplicates/rare interventions comparable on
    # the same base, validation and test observations. Legacy preserves old bytes.
    if generator_version == "independent_v1":
        streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(7)]
        center_rng, base_rng, label_rng, dup_rng, rare_rng, val_rng, test_rng = streams
    else:
        center_rng = base_rng = label_rng = dup_rng = rare_rng = val_rng = test_rng = rng
    d, d_rel, C = 8, 4, 3
    centers = center_rng.normal(0, center_scale, size=(C, d_rel))

    def draw(n_per, shift=0.0, stream=rng):
        Xs, ys = [], []
        for c in range(C):
            rel = centers[c] + stream.normal(0, within_std, size=(n_per, d_rel)) + shift
            noise = stream.normal(0, 1.0, size=(n_per, d - d_rel))
            Xs.append(np.hstack([rel, noise]))
            ys.append(np.full(n_per, c))
        return np.vstack(Xs).astype(np.float32), np.concatenate(ys).astype(np.int64)

    Xb, yb = draw(120, stream=base_rng)
    n_base = len(yb)
    corrupted = np.zeros(n_base, dtype=bool)
    flip = label_rng.choice(n_base, size=int(corruption_rate * n_base), replace=False)
    corrupted[flip] = True
    y_obs = yb.copy()
    for i in flip:
        y_obs[i] = label_rng.choice([c for c in range(C) if c != yb[i]])
    pool = np.nonzero(~corrupted)[0] if generator_version == "legacy" else np.arange(n_base)
    dup_src = dup_rng.choice(pool, size=duplicate_count, replace=False)
    Xd = Xb[dup_src] + dup_rng.normal(0, 0.02, size=(duplicate_count, d)).astype(np.float32)
    yd = y_obs[dup_src]
    rare_center = centers[2] + 9.0
    Xr = np.hstack([rare_center + rare_rng.normal(0, 0.5, size=(rare_count, d_rel)), rare_rng.normal(0, 1, size=(rare_count, d - d_rel))]).astype(np.float32)
    yr = np.full(rare_count, 2, dtype=np.int64)
    X_train = np.vstack([Xb, Xd, Xr])
    y_train = np.concatenate([y_obs, yd, yr])
    truth = {
        "corrupted": np.concatenate([corrupted, corrupted[dup_src], np.zeros(rare_count, dtype=bool)]),
        "duplicate": np.concatenate([np.zeros(n_base, dtype=bool), np.ones(duplicate_count, dtype=bool), np.zeros(rare_count, dtype=bool)]),
        "rare": np.concatenate([np.zeros(n_base + duplicate_count, dtype=bool), np.ones(rare_count, dtype=bool)]),
    }
    Xv, yv = draw(30, stream=val_rng)
    Xt, yt_ = draw(60, stream=test_rng)
    Xs, ys_ = draw(20, shift=shift, stream=test_rng)
    Xrt = np.hstack([rare_center + test_rng.normal(0, 0.5, size=(20, d_rel)), test_rng.normal(0, 1, size=(20, d - d_rel))]).astype(np.float32)
    X_test = np.vstack([Xt, Xs, Xrt])
    y_test = np.concatenate([yt_, ys_, np.full(20, 2)])
    groups = {
        "in_domain": np.concatenate([np.ones(len(yt_), bool), np.zeros(len(ys_) + 20, bool)]),
        "shifted": np.concatenate([np.zeros(len(yt_), bool), np.ones(len(ys_), bool), np.zeros(20, bool)]),
        "rare": np.concatenate([np.zeros(len(yt_) + len(ys_), bool), np.ones(20, bool)]),
    }
    def boundary(X):
        distances = np.linalg.norm(X[:, None, :d_rel] - centers[None, :, :], axis=2)
        nearest = np.sort(distances, axis=1)[:, :2]
        return nearest[:, 1] - nearest[:, 0] <= boundary_margin
    truth["boundary"] = boundary(X_train) & ~truth["rare"]
    groups["boundary"] = boundary(X_test) & ~groups["rare"]
    scaler = StandardScaler().fit(X_train)
    f = lambda a: torch.tensor(scaler.transform(a), dtype=torch.float32)  # noqa: E731
    return T1Data(
        name=name,
        task_type="classification",
        X_train=f(X_train),
        y_train=torch.tensor(y_train),
        X_val=f(Xv),
        y_val=torch.tensor(yv),
        X_test=f(X_test),
        y_test=torch.tensor(y_test, dtype=torch.long),
        num_classes=C,
        case_truth=truth,
        test_groups=groups,
        meta={
            "source": "synthetic_diag",
            "relevant_features": list(range(d_rel)),
            "split_seed": seed,
            "true_labels": np.concatenate([yb, yb[dup_src], yr]).tolist(),
            "synthetic_parameters": {"generator_version": generator_version,
                "center_scale": center_scale, "within_std": within_std,
                "corruption_rate": corruption_rate, "duplicate_count": duplicate_count,
                "rare_count": rare_count, "shift": shift, "boundary_margin": boundary_margin},
            "corrupted_count": int(truth["corrupted"].sum()),
            "duplicate_source_ids": dup_src.tolist(),
            "boundary_definition": "two nearest raw relevant-space center distances differ <= boundary_margin; rare excluded",
            "synthetic_split_policy": "fixed 360 base training, 90 validation, 180 in-domain/60 shifted/20 rare test; splits fractions and max_train do not apply",
        },
    )


def make_nominal_synthetic(seed: int) -> T1Data:
    """Grouped-category engineering task, not a competitive benchmark.

Color and shape are deliberately outside X; size is explicitly represented.
Some observations lack color, making their latent target partly unobservable.
Held-out amber is absent from the training vocabulary. Report those groups.
"""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    def draw(n, rng, heldout):
        color = rng.integers(0, 3, n)
        shape = rng.integers(0, 2, n)
        size = rng.integers(0, 2, n)
        colors = np.asarray(["red", "green", "blue"], dtype=object)[color]
        unknown = np.zeros(n, dtype=bool)
        if heldout:
            unknown[:n // 5] = True
            color[unknown], colors[unknown] = 3, "amber"
        missing = np.zeros(n, dtype=bool)
        missing[-n // 10:] = True
        colors[missing] = None
        y = (color + shape) % 3
        X = np.column_stack([size, rng.normal(size=(n, 2))]).astype(np.float32)
        values = {"color": colors.tolist(), "shape": np.asarray(["circle", "square"])[shape].tolist(),
                  "size": np.asarray(["small", "large"])[size].tolist()}
        return X, y, values, {"known": ~(unknown | missing), "unknown": unknown, "missing": missing}
    Xtr, ytr, ntr, _ = draw(180, streams[0], False)
    Xv, yv, nv, _ = draw(60, streams[1], True)
    Xt, yt, nt, groups = draw(90, streams[2], True)
    scaler = StandardScaler().fit(Xtr)
    tensor = lambda X: torch.tensor(scaler.transform(X), dtype=torch.float32)
    return T1Data(name="synthetic_nominal", task_type="classification",
        X_train=tensor(Xtr), y_train=torch.tensor(ytr, dtype=torch.long),
        X_val=tensor(Xv), y_val=torch.tensor(yv, dtype=torch.long),
        X_test=tensor(Xt), y_test=torch.tensor(yt, dtype=torch.long), num_classes=3,
        nominal_train=ntr, nominal_val=nv, nominal_test=nt, test_groups=groups,
        meta={"source": "synthetic_nominal/v1", "split_seed": seed,
              "nominal_coverage": {"color": False, "shape": False, "size": True},
              "heldout_only_category": "amber", "size_representation_column": 0,
              "target_rule": "(latent color code + shape code) mod 3; missing hides latent color",
              "split_policy": "fixed 180 train/60 validation/90 test; split fractions and max_train do not apply"})
