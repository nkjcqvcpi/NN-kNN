"""Regression tests for the three fixes of 2026-11-26 (after the handoff packs p0-p4).

1. ``train_synchronized`` probed the feature size with gradients enabled. In eval mode that call writes the case
   features into the model's evaluation cache, and the cache treats an all-zero feature row (a dead-ReLU case of an mlp
   encoder) as not cached, so the probe recomputed it under autograd and left the whole cache attached to a stale graph.
   The adapter warm-up (eval mode, frozen retrieval) then backpropagated through it twice: "Trying to backward through
   the graph a second time" (all-case F, classification wine s50).
2. ``ActivationEngine`` picked pre-top-k members among exactly tied scores by its own (numpy) rule, while
   ``model.retrieve`` keeps what ``torch.topk`` keeps: the build-time check failed ("Activation engine disagrees with
   model.retrieve by 0.332", newpol classification wine s32 on xps8960).
3. Opt-in ``statistics.initial_bias: build_time`` seeds the case statistics store with the build-time bias, so
   ``bias_source: delta_from_init`` is no longer b - b = 0 (pack p0 item a).
"""
import os
from pathlib import Path

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import numpy as np
import pytest
import torch
import yaml

from model.common.core import CoreConfig, build_model
from model.t1.baselines import ActivationEngine
from model.t1.calibration import calibrate_free_radius
from model.t1.data import T1Data
from model.t1.outcomes import final_prediction
from model.t1.provenance import CaseStatisticsStore, ScoreConfig, active_cohorts, score_cases, seed_initial_bias
from model.t1.sync import SyncConfig, train_synchronized

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- 1. adapter warm-up after a dead-ReLU case


def dead_relu_core():
    """An mlp core in eval mode whose case 0 has an all-zero feature row (zero input, zero layer biases), with the
    evaluation cache filled without gradients, as the runner leaves it (StabilityTracker snapshot)."""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(40, 3, generator=g)
    X[0] = 0.0
    y = (X[:, 0] > 0).long()
    data = T1Data(name="tiny", task_type="classification", X_train=X, y_train=y, X_val=X[:10] + .05, y_val=y[:10],
                  X_test=X[10:20] - .05, y_test=y[10:20], num_classes=2)
    cc = CoreConfig(task_type="classification", top_k=5, representation="mlp", pre_topk_mask=False, batch_size=8)
    model = build_model(X, y, cc, 2)
    with torch.no_grad():
        for m in model.feature_extractor.modules():
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                m.bias.zero_()
    model._invalidate_case_cache()
    model.eval()
    with torch.no_grad():
        model.retrieve(X[:2], exclude_identical=False)
    assert (model.cached_features[0] == 0).all() and not model.cached_features.requires_grad
    return model, data, cc


def test_adapter_warmup_does_not_backpropagate_through_the_case_cache():
    model, data, cc = dead_relu_core()
    fr, _ = calibrate_free_radius(model, s_task=.5, snapshot_step=0)
    cfg = SyncConfig("joint_rr", 1., .5, .1, 1., 1., .1, epochs=4, patience=4, hidden_dims=(8, 4),
                     joint_objective="summed", adapter_warmup_epochs=2, diagnostics_every=0)
    _, info = train_synchronized(model, data, cc, cfg, fr, near_scale=1.)  # raised RuntimeError before the fix
    assert [h["phase"] for h in info["history"]][:3] == ["adapter", "adapter", "joint"]


def test_the_feature_size_probe_leaves_no_graph_on_the_case_cache():
    model, data, cc = dead_relu_core()
    seen = []
    original = model._extract_features

    def spy(case_indices):
        out = original(case_indices)
        seen.append(model.cached_features is not None and model.cached_features.requires_grad)
        return out

    model._extract_features = spy
    fr, _ = calibrate_free_radius(model, s_task=.5, snapshot_step=0)
    cfg = SyncConfig("joint_rr", 1., .5, .1, 1., 1., .1, epochs=2, patience=2, hidden_dims=(8, 4),
                     adapter_warmup_epochs=1, diagnostics_every=0)
    train_synchronized(model, data, cc, cfg, fr, near_scale=1.)
    assert seen and not any(seen)


# --------------------------------------------------------------------------- 2. engine tie rule = model.retrieve


def tied_core():
    """Cases 0 and 1 are duplicates with different labels and equal biases: their scores tie exactly at the first
    (top_k = 1) place. torch.topk keeps case 1, the engine's own (score, index) rule case 0."""
    X = torch.tensor([[1.0, 1.0], [1.0, 1.0], [5.0, 5.0], [-5.0, 5.0]])
    y = torch.tensor([0, 1, 0, 1])
    model = build_model(X, y, CoreConfig(task_type="classification", top_k=1, pre_topk_mask=True, tau=.5, seed=0), 2)
    with torch.no_grad():
        model.biases.fill_(0.0)
    return model, torch.tensor([[0.0, 0.0], [0.2, 0.1]]), torch.tensor([0, 0])


def model_prediction(model, Xq, mask=None):
    model.eval()
    cm = None if mask is None else torch.as_tensor(mask)
    with torch.no_grad():
        r = model.retrieve(Xq, exclude_identical=False, case_mask=cm)
        _, p, _ = final_prediction(model, Xq, r, case_mask=cm)
    return r, p.double().numpy().reshape(len(Xq), -1)


def test_engine_keeps_the_tied_member_model_retrieve_keeps():
    model, Xq, yq = tied_core()
    r, p = model_prediction(model, Xq)
    kept = [np.flatnonzero(w > 0).tolist() for w in r["weights"].numpy()]
    eng = ActivationEngine(model, Xq, yq, None)  # raised "disagrees with model.retrieve by 1" before the fix
    assert eng.check_max_abs_diff < 1e-6
    members, _ = eng.weights(np.ones(4, dtype=bool), np.arange(2))
    assert [[m for m in row if m >= 0] for row in members.tolist()] == kept
    for mask in ([True, True, False, True], [True, True, True, False], [False, True, True, True]):
        mask = np.array(mask)
        assert np.abs(eng.predict(mask) - model_prediction(model, Xq, mask)[1]).max() < 1e-6


def test_engine_without_ties_is_unchanged():
    """No tie at the k-th place: the model's selection equals the engine's own one, so neighborhoods (members, scores,
    next case) are exactly what the engine computed before, under any case mask."""
    g = torch.Generator().manual_seed(3)
    X = torch.randn(40, 3, generator=g)
    y = (X[:, 0] > 0).long() + (X[:, 1] > .5).long()
    model = build_model(X, y, CoreConfig(task_type="classification", top_k=4, pre_topk_mask=True, tau=.7, seed=0), 3)
    with torch.no_grad():
        model.biases.copy_(torch.randn(40, generator=g) * .3)
    eng = ActivationEngine(model, X, y, torch.arange(40))
    own = ActivationEngine.__new__(ActivationEngine)
    own.__dict__.update(eng.__dict__)
    own.Z_select = None  # the engine's own rule, as before the fix
    rng = np.random.default_rng(0)
    rows = np.arange(40)
    for _ in range(20):
        mask = rng.random(40) > .3
        for a, b in zip(eng.neighborhoods(mask, rows), own.neighborhoods(mask, rows)):
            assert np.array_equal(a, b)


# --------------------------------------------------------------------------- 3. opt-in build-time initial bias


def test_build_time_initial_bias_makes_delta_from_init_nonzero():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(30, 3, generator=g)
    y = (X[:, 0] > 0).long()
    model = build_model(X, y, CoreConfig(task_type="classification", top_k=4, pre_topk_mask=True, seed=0), 2)
    b0 = model.t1_bias_init["value"]
    with torch.no_grad():
        model.biases.add_(torch.randn(30, generator=g) * .2)  # as core training leaves them
    ids = model.active_case_ids().numpy()
    cohorts = active_cohorts(model)
    biases = model.biases[:30].detach().numpy()
    cfg = ScoreConfig(smoothing=1., min_retrieval_count=2, min_activation_mass=.1, bias_source="delta_from_init")
    first_audit = score_cases(ids, cohorts, biases, CaseStatisticsStore("classification"), cfg)
    assert np.all(first_audit.B == first_audit.B[0])  # b - b = 0 everywhere: the recorded behaviour
    store = CaseStatisticsStore("classification")
    assert seed_initial_bias(store, model, "first_audit") == 0 and not list(store.items())
    assert seed_initial_bias(store, model, "build_time") == 30
    assert {st.initial_bias for _, st in store.items()} == {b0}
    seeded = score_cases(ids, cohorts, biases, store, cfg)
    current = score_cases(ids, cohorts, biases, CaseStatisticsStore("classification"),
                          ScoreConfig(smoothing=1., min_retrieval_count=2, min_activation_mass=.1))
    assert len(np.unique(seeded.B)) > 1
    # build_model gives every case one bias, so b - b_build is a constant shift: percentiles equal bias_source current
    assert np.array_equal(seeded.B, current.B)
    with pytest.raises(ValueError, match="initial_bias"):
        seed_initial_bias(store, model, "trained")


def test_p0a_configs_differ_from_the_btransform_configs_only_where_declared():
    src = REPO / "configs/t1/handoff_20261125b"
    dst = REPO / "configs/t1/handoff_20261126/p0a"
    files = sorted(p.name for p in dst.glob("*.yaml"))
    assert files == [f"deltainit_{n}_{r}.yaml" for n in ("medianmad", "percentile") for r in ("classification", "regression")]
    for name in files:
        text = (dst / name).read_text()
        assert "exploratory, our choice, not a PI decision" in text.splitlines()[0]
        tag, role = name[len("deltainit_"):-len(".yaml")].split("_")
        cfg, ref = yaml.safe_load(text), yaml.safe_load((src / f"bnorm_medianmad_{role}.yaml").read_text())
        assert cfg["statistics"] == {**ref["statistics"], "bias_source": "delta_from_init", "initial_bias": "build_time",
                                     "bias_normalization": {"percentile": "within_cohort_percentile",
                                                            "medianmad": "median_mad"}[tag]}
        assert cfg["experiment"] == f"deltainit_{tag}__{role}" and cfg["status"] == "exploratory_handoff_20261126_p0a"
        assert cfg["seeds"] == [30, 31, 32] and cfg["type"] == "retention"
        rest = lambda c: {k: v for k, v in c.items() if k not in ("statistics", "experiment", "status")}  # noqa: E731
        assert rest(cfg) == rest(ref)


def test_initial_bias_key_is_refused_outside_retention(tmp_path, monkeypatch):
    import sys
    from tools import t1_run
    cfg = yaml.safe_load((REPO / "configs/t1/handoff_20261126/p0a/deltainit_percentile_classification.yaml").read_text())
    cfg["type"] = "sync"
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(runs))
    monkeypatch.setattr(sys, "argv", ["t1_run.py", str(path), "--out", str(tmp_path / "out"), "--study", "t"])
    with pytest.raises(SystemExit):
        t1_run.main()
