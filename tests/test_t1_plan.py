"""Unit/integration tests required by the T1 plan (section 15) and the reuse spec (5.5)."""

from __future__ import annotations

import copy
import math
import os

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import numpy as np
import pytest
import torch

from model.nn_cdh import ClassificationNNCDHAdapter
from model.t1.calibration import calibrate_free_radius, l_small
from model.common.core import CoreConfig, build_model, evaluate, make_optimizer, train_retrieval
from model.t1.data import make_splits, make_synthetic
from model.t1.maintenance import CaseArchive, realign_optimizer_state, run_maintenance
from model.t1.provenance import (
    CaseStatisticsStore,
    ScoreConfig,
    active_cohorts,
    audit_provenance,
    score_cases,
    smoothed_quality,
    within_cohort_percentile,
)
from model.t1.retention import RetentionConfig, select_active_set
from model.t1.reuse import ReuseConfig, neighborhood_inputs, train_classification_adapter


def _toy(seed=0, n_per=20):
    g = torch.Generator().manual_seed(seed)
    X = torch.cat([torch.randn(n_per, 3, generator=g) + 3, torch.randn(n_per, 3, generator=g) - 3])
    y = torch.cat([torch.zeros(n_per, dtype=torch.long), torch.ones(n_per, dtype=torch.long)])
    return X, y


def _model(X, y, **over):
    cc = CoreConfig(task_type="classification", epochs=3, patience=5, seed=0, **over)
    return build_model(X, y, cc, 2), cc


SCFG = ScoreConfig(smoothing=1.0, min_retrieval_count=1, min_activation_mass=0.1)


# ------------------------------------------------------------------ provenance / trust


def test_zero_exposure_quality_is_half():
    assert smoothed_quality(0.0, 0.0, 1.0) == pytest.approx(0.5)


def test_quality_moves_with_support():
    assert smoothed_quality(5.0, 0.0, 1.0) > 0.5 > smoothed_quality(0.0, 5.0, 1.0)


def test_superseded_combination_is_rejected():
    with pytest.raises(ValueError, match="supersedes"):
        RetentionConfig(policy="trust", min_per_cohort=0).validate()


def test_percentile_ties_are_neutral():
    v = within_cohort_percentile(np.zeros(6), ["a"] * 3 + ["b"] * 3)
    assert np.allclose(v, 0.5)


def test_audit_is_leave_one_out_and_counts_support():
    X, y = _toy()
    model, _ = _model(X, y)
    store = CaseStatisticsStore()
    audit_provenance(model, X, y, store, exclude_identical=True)
    # well separated classes: all neighbours share the label, so harmful support ~ 0
    H = sum(st.harmful_support for _, st in store.items())
    C = sum(st.positive_support for _, st in store.items())
    assert C > 0 and H < 1e-3 * C
    # without LOO each case would retrieve itself; with LOO total activation mass == #queries
    A = sum(st.activation_mass for _, st in store.items())
    assert A == pytest.approx(X.shape[0], rel=1e-4)


def test_counterfactual_delta_matches_explicit_masking():
    X, y = _toy(n_per=8)
    model, _ = _model(X, y)
    store = CaseStatisticsStore()
    q = X[:3]
    r = model.retrieve(q, exclude_identical=True)
    from model.t1.provenance import _counterfactual_delta

    labels = model.labels[: model.case_count()].float()
    delta = _counterfactual_delta(model, r, y[:3], labels)
    base_p = (r["weights"] @ labels)
    for i in range(model.case_count()):
        if not (r["weights"][:, i] > 0).any():
            continue
        mask = torch.ones(model.case_count(), dtype=torch.bool)
        mask[i] = False
        rw = model.retrieve(q, exclude_identical=True, case_mask=mask)["weights"]
        p = rw @ labels
        for b in range(3):
            if r["weights"][b, i] > 0:
                want = -math.log(max(float(p[b, y[b]]), 1e-8)) + math.log(max(float(base_p[b, y[b]]), 1e-8))
                assert float(delta[b, i]) == pytest.approx(want, abs=1e-4)


def test_statistics_only_mode_does_not_change_behavior():
    X, y = _toy()
    model, _ = _model(X, y)
    before = evaluate(model, X, y)["prob_pre"].clone()
    rng_state = torch.random.get_rng_state()
    audit_provenance(model, X, y, CaseStatisticsStore(), exclude_identical=True)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.allclose(evaluate(model, X, y)["prob_pre"], before)


# ------------------------------------------------------------------ retention / compaction


def test_protected_coverage_cannot_be_violated():
    X, y = _toy()
    model, _ = _model(X, y)
    store = CaseStatisticsStore()
    audit_provenance(model, X, y, store, exclude_identical=True)
    ids = model.active_case_ids().numpy()
    sc = score_cases(ids, active_cohorts(model), model.biases[:40].detach().numpy(), store, SCFG)
    # a policy that would rank every class-1 case last
    sc.bias[:] = np.where(np.array(sc.cohorts) == "class_1", -10.0, 10.0)
    res = select_active_set(sc, 10, RetentionConfig(policy="bias_current", min_per_cohort=3), None)
    kept_cohorts = [sc.cohorts[j] for j in res.keep_slots]
    assert kept_cohorts.count("class_1") >= 3 and len(res.keep_slots) == 10


def test_deterministic_selection():
    X, y = _toy()
    model, _ = _model(X, y)
    store = CaseStatisticsStore()
    audit_provenance(model, X, y, store, exclude_identical=True)
    ids = model.active_case_ids().numpy()
    sc = score_cases(ids, active_cohorts(model), model.biases[:40].detach().numpy(), store, SCFG)
    D = np.random.default_rng(0).random((40, 40))
    cfg = RetentionConfig(policy="kcenter", min_per_cohort=2, seed=3)
    a = select_active_set(sc, 15, cfg, D).keep_slots
    b = select_active_set(sc, 15, copy.deepcopy(cfg), D).keep_slots
    assert np.array_equal(a, b)


def test_compaction_preserves_ids_state_and_optimizer_alignment():
    X, y = _toy()
    model, cc = _model(X, y)
    opt = make_optimizer(model, cc)
    out = model(X[:8])
    torch.nn.functional.nll_loss(torch.log(out[0] + 1e-8), y[:8]).backward()
    opt.step()
    with torch.no_grad():
        model.biases[:40] = torch.arange(40, dtype=torch.float32)
    exp_avg_before = opt.state[model.biases]["exp_avg"].clone()
    keep = np.array([3, 7, 11, 30])
    ids_before = model.case_ids[keep].clone()
    labels_before = model.labels[keep].clone()
    gw_before = model.glocal_weights[keep].clone()
    model.compact_cases(torch.as_tensor(keep))
    realign_optimizer_state(opt, model, keep, 40)
    assert torch.equal(model.case_ids[:4], ids_before)
    assert torch.equal(model.labels[:4], labels_before)
    assert torch.equal(model.glocal_weights[:4], gw_before)
    assert torch.equal(model.biases[:4].detach(), torch.tensor([3.0, 7.0, 11.0, 30.0]))
    assert torch.equal(opt.state[model.biases]["exp_avg"][:4], exp_avg_before[keep])
    assert float(opt.state[model.biases]["exp_avg"][4:40].abs().sum()) == 0.0


def test_archive_restore_reproduces_case_state():
    X, y = _toy()
    model, _ = _model(X, y)
    with torch.no_grad():
        model.biases[:40] = torch.linspace(-1, 1, 40)
    store = CaseStatisticsStore()
    audit_provenance(model, X, y, store, exclude_identical=True)
    archive = CaseArchive()
    snap = {cid: (float(model.biases[i]), model.cases[i].clone()) for i, cid in enumerate(model.active_case_ids().tolist())}
    run_maintenance(model, store, archive, 25, RetentionConfig(policy="bias_current", min_per_cohort=2), SCFG, step=1, run_id="t")
    assert model.case_count() == 25 and len(archive) == 15
    gone = list(archive.entries)[:3]
    model.restore_ids = archive.restore(model, gone)
    ids = model.active_case_ids().tolist()
    for cid in gone:
        s = ids.index(cid)
        assert float(model.biases[s]) == pytest.approx(snap[cid][0])
        assert torch.equal(model.cases[s], snap[cid][1])


def test_test_data_never_affects_retention():
    data = make_synthetic("synthetic_diag", 0)
    cc = CoreConfig(task_type="classification", epochs=2, seed=0)
    m1 = build_model(data.X_train, data.y_train, cc, 3)
    m2 = copy.deepcopy(m1)
    s1, s2 = CaseStatisticsStore(), CaseStatisticsStore()
    audit_provenance(m1, data.X_train, data.y_train, s1, exclude_identical=True)
    evaluate(m2, data.X_test, data.y_test)  # touching test data must not change statistics
    audit_provenance(m2, data.X_train, data.y_train, s2, exclude_identical=True)
    for cid, st in s1.items():
        assert st.positive_support == pytest.approx(s2.get(cid).positive_support)


# ------------------------------------------------------------------ free-correction radius


def test_free_radius_excludes_self_and_penalty_shape():
    X, y = _toy()
    model, _ = _model(X, y)
    fr, _ = calibrate_free_radius(model, s_task=1.0, snapshot_step=0)
    assert fr.n_possible_pairs == 40 * 39
    c = torch.tensor([0.0, fr.tau_task, fr.tau_task + 0.5, fr.tau_task + 1.0])
    vals = [float(l_small(c[i : i + 1], fr)) for i in range(4)]
    assert vals[0] == 0.0 and vals[1] == 0.0 and 0 < vals[2] < vals[3]


def test_free_radius_degenerate_is_stable():
    X, y = _toy()
    model, _ = _model(X, y)
    with torch.no_grad():
        model.biases.fill_(-1e6)
    fr, _ = calibrate_free_radius(model, s_task=1.0, snapshot_step=0)
    assert fr.status == "empty_region" and fr.tau_task == 0.0
    assert math.isfinite(float(l_small(torch.tensor([0.3]), fr)))


# ------------------------------------------------------------------ reuse (plan 5.5)


def test_adapter_zero_correction_identity_and_dims():
    ad = ClassificationNNCDHAdapter(3, 4, 0, (8, 4), "nominal_residual_scores")
    for p in ad.net[-1].parameters():
        torch.nn.init.zeros_(p)
    p0 = torch.softmax(torch.randn(5, 4), 1)
    r, s = ad(torch.randn(5, 3), p0, None)
    assert s.shape == (5, 4) and torch.allclose(s, p0)
    ad2 = ClassificationNNCDHAdapter(3, 4, 0, (8, 4), "logit_residual")
    for p in ad2.net[-1].parameters():
        torch.nn.init.zeros_(p)
    _, pf = ad2(torch.randn(5, 3), p0, None)
    assert torch.allclose(pf, p0, atol=1e-6)


def test_adapter_input_has_no_raw_query_or_neighborhood_embedding():
    X, y = _toy()
    model, _ = _model(X, y)
    nb = neighborhood_inputs(model, X[:4], exclude_identical=True)
    assert set(k for k, v in nb.items() if v is not None) == {"dz", "p0", "weights"}
    ad = ClassificationNNCDHAdapter(3, 2, 0, (8, 4))
    assert ad.net[0].in_features == 3 + 2  # [Delta_z, p0] only


def test_class_permutation_equivariance_of_retrieval():
    X, y = _toy()
    m1, _ = _model(X, y)
    m2, _ = _model(X, 1 - y)
    p1 = evaluate(m1, X, y)["prob_pre"]
    p2 = evaluate(m2, X, 1 - y)["prob_pre"]
    assert torch.allclose(p1, p2.flip(1), atol=1e-6)


def test_disabled_adapter_does_not_change_statistics():
    X, y = _toy()
    model, _ = _model(X, y)
    s1, s2 = CaseStatisticsStore(), CaseStatisticsStore()
    audit_provenance(model, X, y, s1, exclude_identical=True)
    model.classification_adapter = ClassificationNNCDHAdapter(3, 2, 0, (8, 4))
    model.enable_classification_adapter = False
    audit_provenance(model, X, y, s2, exclude_identical=True)
    for cid, st in s1.items():
        assert st.positive_support == pytest.approx(s2.get(cid).positive_support)


def test_disabled_path_equivalence_forward():
    X, y = _toy()
    model, _ = _model(X, y)
    model.eval()
    a = model(X)[0]
    model.classification_adapter = ClassificationNNCDHAdapter(3, 2, 0, (8, 4))
    model.enable_classification_adapter = False
    b = model(X)[0]
    assert torch.allclose(a, b)


def test_checkpoint_reload_preserves_cases_and_ids(tmp_path):
    X, y = _toy()
    model, cc = _model(X, y)
    store = CaseStatisticsStore()
    audit_provenance(model, X, y, store, exclude_identical=True)
    run_maintenance(model, store, CaseArchive(), 20, RetentionConfig(policy="random", min_per_cohort=2, seed=1), SCFG, step=0, run_id="t")
    path = tmp_path / "m.pt"
    torch.save({"state": model.state_dict(), "active": model.case_count(), "stats": store.state_dict()}, path)
    m2, _ = _model(X, y)
    ck = torch.load(path)
    m2.load_state_dict(ck["state"])
    m2.set_active_case_count(ck["active"])
    assert torch.equal(m2.active_case_ids(), model.active_case_ids())
    assert torch.allclose(evaluate(m2, X, y)["prob_pre"], evaluate(model, X, y)["prob_pre"])
    assert CaseStatisticsStore.from_state_dict(ck["stats"]).get(0).audits == 1


def test_end_to_end_training_and_reuse_smoke():
    data = make_synthetic("synthetic_diag", 1)
    cc = CoreConfig(task_type="classification", epochs=4, patience=5, seed=1)
    model = build_model(data.X_train, data.y_train, cc, 3)
    tr = train_retrieval(model, data.X_train, data.y_train, data.X_val, data.y_val, cc)
    assert tr.epochs_run >= 1 and float(model.biases[: model.case_count()].std()) > 0  # biases actually train
    rc = ReuseConfig(output_mode="nominal_residual_scores", loss="combined", lambda_diff=1, lambda_cls=1, probability_mode="softmax", epochs=3)
    ad, info = train_classification_adapter(model, data, rc)
    assert info["neighborhoods"] == "leave_one_out"


def test_regression_counterfactual_audit_runs():
    data = make_splits("energy_efficiency", "regression", 0, test_frac=0.2, val_frac=0.2)
    cc = CoreConfig(task_type="regression", epochs=1, seed=0)
    model = build_model(data.X_train[:60], data.y_train[:60], cc, None)
    store = CaseStatisticsStore("regression")
    audit_provenance(model, data.X_train[:60], data.y_train[:60], store, exclude_identical=True, reg_bins=data.reg_bins, regression_success_tolerance=0.5)
    tot = sum(st.positive_support + st.harmful_support for _, st in store.items())
    assert tot > 0


def test_compaction_never_mutates_caller_training_data():
    X, y = _toy()
    X0, y0 = X.clone(), y.clone()
    model, _ = _model(X, y)
    model.compact_cases(torch.tensor([5, 1, 30]))
    assert torch.equal(X, X0) and torch.equal(y, y0)


def test_clone_optimizer_continues_state():
    from model.common.core import clone_optimizer

    X, y = _toy()
    model, cc = _model(X, y)
    tr = train_retrieval(model, X, y, X, y, cc, epochs=2)
    m2 = copy.deepcopy(model)
    opt2 = clone_optimizer(tr.optimizer, m2, cc)
    a = tr.optimizer.state[model.biases]["exp_avg_sq"]
    b = opt2.state[m2.biases]["exp_avg_sq"]
    assert torch.equal(a, b) and a.abs().sum() > 0


def test_finetune_include_initial_never_worse_on_validation():
    X, y = _toy()
    model, cc = _model(X, y)
    tr = train_retrieval(model, X, y, X, y, cc, epochs=3)
    v0 = evaluate(model, X, y)["loss_pre"]
    from model.common.core import clone_optimizer

    opt = clone_optimizer(tr.optimizer, model, cc)
    with torch.no_grad():
        opt.param_groups[0]["lr"] = 10.0  # absurd step: fine-tuning should be rejected
    train_retrieval(model, X, y, X, y, cc, epochs=2, optimizer=opt, include_initial=True)
    assert evaluate(model, X, y)["loss_pre"] <= v0 + 1e-9
