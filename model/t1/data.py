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
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    if task_type == "classification":
        from datasets.classification_data import load_small_classification_dataset

        st = load_small_classification_dataset(name, seed=seed)
        return st["X"].numpy().astype(np.float32), st["y"].numpy().astype(np.int64), {"source": "datasets.classification_data"}
    from datasets.reg_data import Reg_data

    X, y = Reg_data(name)
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32).reshape(-1),
        {"source": "datasets.reg_data"},
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
) -> T1Data:
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
    tr_idx, te_idx = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=strat)
    strat_tr = y[tr_idx] if task_type == "classification" else None
    rel_val = val_frac / (1.0 - test_frac)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=rel_val, random_state=seed + 1, stratify=strat_tr)
    mt_idx = None
    if maint_frac > 0:
        strat_tr = y[tr_idx] if task_type == "classification" else None
        rel_m = maint_frac / (1.0 - test_frac - val_frac)
        tr_idx, mt_idx = train_test_split(tr_idx, test_size=rel_m, random_state=seed + 2, stratify=strat_tr)
    if max_train is not None and len(tr_idx) > max_train:
        strat_tr = y[tr_idx] if task_type == "classification" else None
        tr_idx, _ = train_test_split(tr_idx, train_size=max_train, random_state=seed + 3, stratify=strat_tr)
    scaler = StandardScaler().fit(X[tr_idx])
    t = lambda a: torch.tensor(scaler.transform(X[a]), dtype=torch.float32)  # noqa: E731
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
    return data


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
