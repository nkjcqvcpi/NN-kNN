"""Checks added by the 2026-11-24 progress review (reports/2026-11-24-progress-review.md).

One fixed defect (t1_summarize counted seeds the wrong way for lower-is-better metrics) and three demonstrated
problems that are properties of the recorded design, not small code bugs. Those are strict xfail tests: they fail
today, and they turn into errors (XPASS) when the design changes, so the change is noticed. Fixing them here would
change the meaning of units already recorded (splits, evidence rule), which the owner's rules forbid.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

os.environ.setdefault("NNKNN_DEVICE", "cpu")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ----------------------------------------------------------------------------- t1_summarize direction (fixed)

def _paired(tmp_path, monkeypatch, rows, *extra):
    from tools import t1_summarize
    monkeypatch.setattr(t1_summarize, "require_unit_tree", lambda p: Path(p))
    monkeypatch.setattr(t1_summarize, "experiment_rows", lambda src: rows)
    argv = ["t1_summarize.py", str(tmp_path / "units"), "--metric", "test_rmse_pre", "--metric", "test_accuracy_pre",
            "--reference", "ref", "--out", str(tmp_path / "out"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    with contextlib.redirect_stdout(io.StringIO()):
        t1_summarize.main()
    return pd.read_csv(tmp_path / "out" / "paired_vs_ref.csv").set_index("metric")


def _rows():
    # the condition has a lower RMSE (better) and a higher accuracy (better) than the reference on both seeds
    return [dict(dataset="d", condition=c, seed=s, test_rmse_pre=r, test_accuracy_pre=a)
            for s in (30, 31) for c, r, a in (("ref", 1.0 + s / 100, 0.80), ("cond", 0.9 + s / 200, 0.85 + s / 1000))]


def test_summarize_counts_lower_error_as_better(tmp_path, monkeypatch):
    p = _paired(tmp_path, monkeypatch, _rows())
    assert p.loc["test_rmse_pre", "mean_diff"] < 0
    assert (p.loc["test_rmse_pre", "seeds_better"], p.loc["test_rmse_pre", "seeds_worse"]) == (2, 0)
    assert (p.loc["test_accuracy_pre", "seeds_better"], p.loc["test_accuracy_pre", "seeds_worse"]) == (2, 0)


def test_summarize_direction_override(tmp_path, monkeypatch):
    p = _paired(tmp_path, monkeypatch, _rows(), "--higher-is-better", "test_rmse_pre")
    assert (p.loc["test_rmse_pre", "seeds_better"], p.loc["test_rmse_pre", "seeds_worse"]) == (0, 2)


def test_metric_direction_from_name():
    from tools.t1_summarize import metric_lower_is_better as low
    for m in ("test_rmse_post", "test_error", "val_loss", "selected_drift", "churn", "test_mae_pre", "nll"):
        assert low(m), m
    for m in ("test_accuracy_pre", "macro_f1", "auc"):
        assert not low(m), m


# ----------------------------------------------------------------------------- evidence rule under all-case softmax

def _evidenced_share(pre_topk_mask: bool) -> float:
    from model.common.core import CoreConfig, build_model
    from model.t1.data import make_splits
    from model.t1.provenance import CaseStatisticsStore, ScoreConfig, active_cohorts, audit_provenance, score_cases
    torch.manual_seed(0)
    d = make_splits("synthetic_diag", "classification", 30, test_frac=.2, val_frac=.2)
    cfg = CoreConfig(task_type="classification", seed=30, representation="raw", top_k=5, pre_topk_mask=pre_topk_mask,
                     bias_init="knn_mean", bias_init_k=5)
    model = build_model(d.X_train, d.y_train, cfg, d.num_classes)
    store = CaseStatisticsStore("classification")
    n = model.case_count()
    audit_provenance(model, d.X_train, d.y_train, store, exclude_identical=True, counterfactual=False,
                     query_case_ids=model.active_case_ids())
    ids = model.active_case_ids().detach().cpu().numpy()
    # the _base.yaml evidence rule: min_retrieval_count 2, min_activation_mass 0.1
    s = score_cases(ids, active_cohorts(model), model.biases[:n].detach().cpu().numpy(), store,
                    ScoreConfig(smoothing=1.0, min_retrieval_count=2, min_activation_mass=0.1))
    return float(np.mean(s.evidenced))


def test_evidence_rule_separates_cases_under_top_k_mask():
    assert _evidenced_share(True) < 1.0


@pytest.mark.xfail(strict=True, reason=(
    "Review 2026-11-24 problem: R_i counts queries with weight > 1e-6; under all-case softmax every case gets such a "
    "weight from (almost) every query, so the _base.yaml rule (min_retrieval_count 2, min_activation_mass 0.1, tuned "
    "for top-5) marks every case evidenced. Retention/provenance claims under the adopted all-case setting need a "
    "redefined rule (a PI decision); the review study review_retention_r760 measures the effect."))
def test_evidence_rule_separates_cases_under_all_case_softmax():
    assert _evidenced_share(False) < 1.0


# ----------------------------------------------------------------------------- split properties of recorded tasks

def _have(name, task_type):
    from model.t1.data import make_splits
    try:
        make_splits(name, task_type, 30, test_frac=.2, val_frac=.2)
    except Exception as exc:  # noqa: BLE001  dataset files are not in git
        pytest.skip(f"dataset {task_type}/{name} not available: {exc!r}")


@pytest.mark.xfail(strict=True, reason=(
    "Review 2026-11-24 problem: regression__wine (prereg primary task) has about 22% of its test rows repeated "
    "exactly (features and target) in the training set (1-NN RMSE 0.000 on them). Changing the split would change "
    "every recorded wine unit; the review reports results with and without these rows instead."))
def test_regression_wine_test_rows_are_not_train_duplicates():
    _have("wine", "regression")
    from model.t1.data import make_splits
    d = make_splits("wine", "regression", 30, test_frac=.2, val_frac=.2)
    train = {tuple(r) + (round(float(t), 5),) for r, t in zip(d.X_train.numpy().round(5).tolist(), d.y_train.numpy().reshape(-1))}
    dup = [tuple(r) + (round(float(t), 5),) in train for r, t in zip(d.X_test.numpy().round(5).tolist(), d.y_test.numpy().reshape(-1))]
    assert np.mean(dup) <= 0.01


@pytest.mark.xfail(strict=True, reason=(
    "Review 2026-11-24 problem: non-OpenML tasks draw a fresh holdout per seed, so most test rows of the evaluation "
    "seeds 40-42 were validation rows (used for model selection) under the development seeds 20-25 (iris 67-77%). "
    "A fixed outer holdout would need a new prereg addendum (PI/owner decision)."))
def test_evaluation_test_rows_were_not_development_validation_rows():
    _have("iris", "classification")
    from model.t1.data import make_splits
    dev_val = set()
    for s in range(20, 26):
        dev_val |= set(make_splits("iris", "classification", s, test_frac=.2, val_frac=.2).meta["split_indices"]["validation"])
    test = make_splits("iris", "classification", 40, test_frac=.2, val_frac=.2).meta["split_indices"]["test"]
    assert np.mean([i in dev_val for i in test]) <= 0.05
