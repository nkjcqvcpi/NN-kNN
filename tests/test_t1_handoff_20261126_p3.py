"""Handoff 2026-11-26 pack p3: learned-distance percentile bias initializer (core, opt-in), the historical per-case
adapter and the direct classification head as opt-in reuse output modes, the adapter without the explicit nominal
channel, and the probability-vector free-radius label metric; plus the pack's config files.

Defaults stay untouched: every new behaviour is reached only through a new config value.
"""
import copy
import os
from pathlib import Path

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from model.common.core import CoreConfig, build_model, kth_neighbor_learned_distance, learned_distance_percentile, \
    retrieval_events
from model.common.geometry import case_case_distance
from model.common.nominal import encoded_queries
from model.common.outcomes import final_prediction
from model.t1.calibration import calibrate_free_radius, loo_class_probabilities
from model.t1.data import T1Data, make_nominal_synthetic
from model.t1.provenance import CaseStatisticsStore, audit_provenance
from model.t1.reuse import (ReuseConfig, evaluate_regression_reuse, evaluate_reuse, neighborhood_inputs,
                            train_classification_adapter, train_regression_adapter)
from model.t1.reuse_variants import (DirectHeadAdapter, PerCaseClassificationAdapter, PerCaseRegressionAdapter,
                                     neighbor_slots)
from model.t1.sync import _forward_terms

REPO = Path(__file__).resolve().parents[1]
P3 = REPO / "configs/t1/handoff_20261126/p3-core-reuse"
FIELDS = [{"name": "color", "covered_by_representation": False},
          {"name": "shape", "covered_by_representation": False},
          {"name": "size", "covered_by_representation": True}]


def toy_classification(seed=0, n=60):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 3, generator=g)
    y = (X[:, 0] > -.4).long() + (X[:, 0] + .3 * X[:, 1] > .4).long()
    a, b = int(n * .6), int(n * .8)
    return T1Data(name="toy", task_type="classification", X_train=X[:a], y_train=y[:a], X_val=X[a:b], y_val=y[a:b],
                  X_test=X[b:], y_test=y[b:], num_classes=3)


def toy_regression(seed=1, n=48):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 2, generator=g)
    y = X[:, 0] - .5 * X[:, 1] ** 2 + .1 * torch.randn(n, generator=g)
    a, b = int(n * .625), int(n * .8125)
    return T1Data(name="toy", task_type="regression", X_train=X[:a], y_train=y[:a], X_val=X[a:b], y_val=y[a:b],
                  X_test=X[b:], y_test=y[b:])


def rc(mode, loss="combined", **kw):
    return ReuseConfig(mode, loss, 1., 1., "softmax", hidden_dims=(8, 4), epochs=kw.pop("epochs", 6),
                       patience=kw.pop("patience", 3), **kw)


def core(data, **kw):
    cc = CoreConfig(task_type=data.task_type, top_k=kw.pop("top_k", 4), seed=3, **kw)
    return build_model(data.X_train, data.y_train, cc, data.num_classes)


# --------------------------------------------------------------------------- (a) percentile initializer


def test_default_initializer_is_unchanged():
    data = toy_classification()
    cc = CoreConfig(task_type="classification", top_k=4)
    assert cc.bias_init == "knn_mean" and cc.bias_init_k == 5
    model = build_model(data.X_train, data.y_train, cc, 3)
    assert set(model.t1_bias_init) == {"rule", "k", "value"}
    assert model.t1_bias_init["value"] == pytest.approx(kth_neighbor_learned_distance(model, 5))
    assert torch.allclose(model.biases, torch.full_like(model.biases, model.t1_bias_init["value"]))


@pytest.mark.parametrize("pct", [0, 10, 25, 50, 100])
def test_percentile_initializer_is_the_percentile_of_all_nonself_learned_distances(pct):
    data = toy_classification()
    model = core(data, bias_init="learned_distance_percentile", bias_init_percentile=pct)
    D = case_case_distance(model).double().numpy()
    off = D[~np.eye(len(D), dtype=bool)]
    expected = np.percentile(off, pct)
    assert model.t1_bias_init == {"rule": "learned_distance_percentile", "k": 5, "value": pytest.approx(expected),
                                  "percentile": float(pct)}
    assert model.case_default_bias == pytest.approx(expected)
    assert torch.allclose(model.biases, torch.full_like(model.biases, float(expected)))
    if pct == 0:  # self pairs excluded: the smallest distance is a non-self one, not 0
        assert expected == pytest.approx(off.min()) and expected > 0


def test_percentile_initializer_uses_the_learned_metric_and_keeps_duplicates():
    data = toy_classification()
    X = torch.cat([data.X_train, data.X_train[:5]])  # five distinct duplicate cases: valid zero distances
    y = torch.cat([data.y_train, data.y_train[:5]])
    model = build_model(X, y, CoreConfig(task_type="classification", representation="mlp", top_k=4,
                                         bias_init="learned_distance_percentile", bias_init_percentile=1), 3)
    D = case_case_distance(model).double().numpy()
    off = D[~np.eye(len(D), dtype=bool)]
    assert (off < 1e-6).sum() == 10
    assert model.t1_bias_init["value"] == pytest.approx(np.percentile(off, 1))
    with pytest.raises(ValueError, match="within"):
        learned_distance_percentile(model, 101)


def test_percentiles_are_monotone_and_differ_from_knn_mean():
    data = toy_classification()
    values = [core(data, bias_init="learned_distance_percentile", bias_init_percentile=p).t1_bias_init["value"]
              for p in (10, 25, 50)]
    assert values[0] < values[1] < values[2]
    assert core(data).t1_bias_init["value"] not in values


# --------------------------------------------------------------------------- (d) probability-vector label metric


def test_loo_class_probabilities_match_training_loo_retrieval():
    data = toy_classification()
    model = core(data)
    with torch.no_grad():
        model.biases.add_(torch.linspace(-.2, .2, len(model.biases)))
    P = loo_class_probabilities(model, chunk=7)
    model.eval()
    r = model.retrieve(data.X_train, exclude_identical=False, query_case_ids=torch.arange(len(data.y_train)))
    p = r["weights"] @ model.labels[r["case_indices"]].float()
    assert torch.allclose(P, p / p.sum(1, keepdim=True), atol=1e-6)
    assert model.training is False  # it was in eval mode before and stays there


def test_probability_vector_metric_is_an_opt_in_ablation():
    data = toy_classification()
    model = core(data)
    default, snap = calibrate_free_radius(model, s_task=.5, snapshot_step=0)
    explicit, _ = calibrate_free_radius(model, s_task=.5, snapshot_step=0, label_metric="one_hot_euclidean")
    assert default == explicit and default.label_metric == "one_hot_euclidean"
    assert "label_probability_snapshot" not in snap
    pv, psnap = calibrate_free_radius(model, s_task=.5, snapshot_step=0, label_metric="probability_vector_distance")
    assert pv.label_metric == "probability_vector_distance" and pv.n_pairs == default.n_pairs
    P = loo_class_probabilities(model)
    assert torch.equal(psnap["label_probability_snapshot"], P)
    dy = (P.unsqueeze(1) - P.unsqueeze(0)).norm(dim=-1)  # exact pairwise differences
    assert pv.tau_task == pytest.approx(float(dy[psnap["pair_mask"]].mean()), rel=1e-6)
    assert 0 <= pv.tau_task <= 2 ** .5 + 1e-6
    med, _ = calibrate_free_radius(model, s_task=.5, snapshot_step=0, label_metric="probability_vector_distance",
                                   aggregation="median")
    assert med.tau_task == pytest.approx(float(dy[psnap["pair_mask"]].median()), rel=1e-6)
    with pytest.raises(ValueError, match="Unknown free-radius label metric"):
        calibrate_free_radius(model, s_task=.5, snapshot_step=0, label_metric="kl")
    reg = toy_regression()
    rmodel = core(reg)
    with pytest.raises(ValueError, match="does not apply"):
        calibrate_free_radius(rmodel, s_task=.5, snapshot_step=0, label_metric="probability_vector_distance")
    a, _ = calibrate_free_radius(rmodel, s_task=.5, snapshot_step=0)
    b, _ = calibrate_free_radius(rmodel, s_task=.5, snapshot_step=0, label_metric="absolute_difference")
    assert a == b


def test_probability_vector_metric_gives_equal_vectors_distance_zero():
    """Cases with equal leave-one-out probability vectors (here: exact duplicates) are at label distance 0, not at
    the ~1e-4 the matrix-product shortcut of torch.cdist leaves on inputs of more than 25 rows."""
    data = toy_classification(n=100)
    X = torch.cat([data.X_train, data.X_train[:20]])
    y = torch.cat([data.y_train, data.y_train[:20]])
    model = build_model(X, y, CoreConfig(task_type="classification", top_k=4, seed=3), 3)
    fr, snap = calibrate_free_radius(model, s_task=.5, snapshot_step=0, label_metric="probability_vector_distance")
    P = snap["label_probability_snapshot"]
    n0 = len(data.y_train)
    same = (P[:20] == P[n0:n0 + 20]).all(1)
    assert same.any()
    exact = (P.unsqueeze(1) - P.unsqueeze(0)).norm(dim=-1)
    assert torch.equal(exact[torch.arange(20)[same], (n0 + torch.arange(20))[same]],
                       torch.zeros(int(same.sum())))
    # the shortcut moves tau by ~2e-7 here; direct differences agree to float32 rounding of the mean
    assert abs(fr.tau_task - float(exact[snap["pair_mask"]].mean())) < 2e-8


# --------------------------------------------------------------------------- (b) per-case adapter


def test_neighbor_slots_keep_every_positive_activation():
    w = torch.tensor([[.5, 0., .5, 0.], [0., 1., 0., 0.]])
    idx, wk = neighbor_slots(w)
    assert idx.shape == (2, 2)
    assert torch.allclose(torch.zeros(2, 4).scatter(1, idx, wk), w)


def test_per_case_classification_targets_follow_the_handoff_sign_convention():
    """One retrieved case of the correct class: target 0; of class j for target k: -1 at j, +1 at k."""
    C = 4
    yc = F.one_hot(torch.tensor([[1], [2]]), C).float()  # [B=2, k=1, C]
    y = torch.tensor([1, 3])
    target = F.one_hot(y, C).float().unsqueeze(1) - yc
    assert torch.equal(target[0, 0], torch.zeros(C))
    assert target[1, 0].tolist() == [0., 0., -1., 1.]


def test_per_case_classification_forward_matches_its_definition():
    data = toy_classification()
    model = core(data)
    torch.manual_seed(0)
    ad = PerCaseClassificationAdapter(3, 3, (8, 4))
    model.eval()
    r = model.retrieve(data.X_test, exclude_identical=False)
    residual, s = ad.forward_retrieval(model, r)
    w, labels = r["weights"], model.labels[r["case_indices"]].float()
    zq, zc = r["query_features"], r["case_features"]
    manual = torch.zeros_like(s)
    for q in range(len(zq)):
        for i in torch.nonzero(w[q] > 0).view(-1):
            ri = torch.tanh(ad.net(torch.cat([zc[i], zq[q] - zc[i]])))
            manual[q] += w[q, i] * (labels[i] + ri)
    assert torch.allclose(s, manual, atol=1e-6)
    p0 = w @ labels
    assert torch.allclose(s - p0, residual, atol=1e-6)


@pytest.mark.parametrize("loss", ["diff_only", "combined", "cls_only"])
def test_per_case_classification_trains_on_frozen_retrieval_and_reaches_every_reader(loss):
    data = toy_classification()
    model = core(data)
    before = copy.deepcopy(model.state_dict())
    cfg = rc("per_case_2022", loss)
    ad, info = train_classification_adapter(model, data, cfg)
    assert isinstance(ad, PerCaseClassificationAdapter) and info["actual_output_mode"] == "per_case_2022"
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert len(info["history"]) == info["epochs_run"] and info["best_epoch"] >= 1
    m = evaluate_reuse(model, ad, data.X_test, data.y_test, cfg, temperature=1.0)
    model.eval()
    r = model.retrieve(data.X_test, exclude_identical=False)
    pre, final, kind = final_prediction(model, data.X_test, r, adapter=ad)
    assert kind == "cross_entropy"
    assert m["accuracy_post"] == pytest.approx(float((final.argmax(1) == data.y_test).float().mean()))
    assert m["accuracy_pre"] == pytest.approx(float((pre.argmax(1) == data.y_test).float().mean()))
    nb = neighborhood_inputs(model, data.X_test, exclude_identical=False, nominal_adapter=ad)
    assert torch.allclose(nb["outputs"][1], final)
    T = _forward_terms(model, ad, data.X_test, data.y_test, None, output_mode="per_case_2022", near_scale=1.)
    assert float(T["L_post"]) == pytest.approx(float(F.cross_entropy(final, data.y_test)), rel=1e-5)
    store = CaseStatisticsStore("classification")
    audit_provenance(model, data.X_train, data.y_train, store, exclude_identical=True, counterfactual=False, step=0,
                     query_case_ids=torch.arange(len(data.y_train)), adapter=ad)
    events = retrieval_events(model, data.X_test, data.y_test, stream="test", run_id="t", adapter=ad)
    assert events[0]["adapter_output_mode"] == "per_case_2022" and events[0]["delta_u"] is None


def test_per_case_nominal_adapter_uses_per_case_nominal_differences():
    data = make_nominal_synthetic(8)
    model = build_model(data.X_train, data.y_train, CoreConfig(task_type="classification", seed=1), 3)
    cfg = rc("per_case_2022", "combined", epochs=2, nominal_fields=FIELDS, nominal_fields_covered_by_representation=False)
    ad, info = train_classification_adapter(model, data, cfg)
    assert ad.nominal_dim == 9 and "nominal_manifest" in info
    q = encoded_queries(ad, data, "test")
    m = evaluate_reuse(model, ad, data.X_test, data.y_test, cfg, query_nominal=q, test_groups=data.test_groups)
    assert m["n_unknown"] == 18 and np.isfinite(m["accuracy_post"])
    model.eval()
    r = model.retrieve(data.X_test[:5], exclude_identical=False)
    _, s = ad.forward_retrieval(model, r, q[:5])
    # manual: the first query's pairs with e(u_q) - e(u_i)
    idx, wk = neighbor_slots(r["weights"][:1])
    ids = model.active_case_ids()[r["case_indices"]][idx[0]]
    uc = ad.values_for_ids(ids)
    zc = r["case_features"][idx[0]]
    zq = r["query_features"][0].expand_as(zc)
    ri = torch.tanh(ad.net(torch.cat([zc, uc, zq - zc, q[0].expand_as(uc) - uc], 1)))
    yc = model.labels[r["case_indices"]].float()[idx[0]]
    assert torch.allclose(s[0], (wk[0, :, None] * (yc + ri)).sum(0), atol=1e-6)
    with pytest.raises(ValueError, match="explicit encoded query"):
        ad.forward_retrieval(model, r, None)


@pytest.mark.parametrize("loss", ["diff_only", "combined"])
def test_per_case_regression_adapter(loss):
    data = toy_regression()
    model = core(data)
    before = copy.deepcopy(model.state_dict())
    cfg = rc("per_case_2022", loss)
    ad, info = train_regression_adapter(model, data, cfg)
    assert isinstance(ad, PerCaseRegressionAdapter) and info["actual_output_mode"] == "per_case_regression_2022"
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert not any(p.requires_grad for p in ad.adapt_net_final.parameters())
    model.eval()
    r = model.retrieve(data.X_test, exclude_identical=False)
    labels = model.labels[r["case_indices"]].float()
    out = ad.forward_aggregate(r["query_features"], r["case_features"], labels, r["weights"])
    full = ad(r["query_features"], r["case_features"], labels)  # the maintained all-pairs per-case forward
    assert torch.allclose(out, (r["weights"].unsqueeze(-1) * full).sum(1), atol=1e-5)
    pre, final, kind = final_prediction(model, data.X_test, r, adapter=ad)
    assert kind == "squared_error" and torch.allclose(final, out)
    m = evaluate_regression_reuse(model, ad, data.X_test, data.y_test, success_tolerance=.5)
    assert m["rmse_post"] == pytest.approx(float(((out.view(-1) - data.y_test) ** 2).mean().sqrt()), rel=1e-5)


# --------------------------------------------------------------------------- (c) direct head


def test_direct_head_is_a_plain_classifier_on_z_q():
    data = toy_classification()
    model = core(data)
    cfg = rc("direct_head", "cls_only")
    ad, info = train_classification_adapter(model, data, cfg)
    assert isinstance(ad, DirectHeadAdapter) and info["actual_output_mode"] == "direct_head"
    model.eval()
    r = model.retrieve(data.X_test, exclude_identical=False)
    _, s = ad.forward_retrieval(model, r)
    assert torch.allclose(s, ad.net(data.X_test), atol=1e-6)  # raw core: z_q is the input itself
    with torch.no_grad():
        model.biases.add_(torch.randn(len(model.biases)))  # retrieval changes, the decision does not
    _, s2 = ad.forward_retrieval(model, model.retrieve(data.X_test, exclude_identical=False))
    assert torch.allclose(s, s2)
    m = evaluate_reuse(model, ad, data.X_test, data.y_test, cfg)
    assert m["accuracy_post"] == pytest.approx(float((s.argmax(1) == data.y_test).float().mean()))
    for bad in ("combined", "diff_only"):
        with pytest.raises(ValueError, match="cls_only"):
            rc("direct_head", bad).validate()
    with pytest.raises(ValueError, match="classification capacity reference"):
        train_regression_adapter(core(toy_regression()), toy_regression(), cfg)


def test_direct_head_with_uncovered_nominal_fields_reads_the_query_groups():
    data = make_nominal_synthetic(9)
    model = build_model(data.X_train, data.y_train, CoreConfig(task_type="classification", seed=1), 3)
    cfg = rc("direct_head", "cls_only", epochs=2, nominal_fields=FIELDS, nominal_fields_covered_by_representation=False)
    ad, _ = train_classification_adapter(model, data, cfg)
    assert ad.query_nominal_dim == 9 and ad.nominal_dim == 0
    q = encoded_queries(ad, data, "test")
    m = evaluate_reuse(model, ad, data.X_test, data.y_test, cfg, query_nominal=q)
    assert np.isfinite(m["accuracy_post"])


# --------------------------------------------------------------------------- (e) without Delta_u_q


def test_omit_nominal_channel_gives_the_adapter_no_delta_u():
    data = make_nominal_synthetic(10)
    model = build_model(data.X_train, data.y_train, CoreConfig(task_type="classification", seed=1), 3)
    cfg = rc("nominal_residual_scores", "combined", epochs=2, nominal_fields=FIELDS,
             nominal_fields_covered_by_representation=False, omit_nominal_channel=True)
    ad, info = train_classification_adapter(model, data, cfg)
    assert ad.nominal_dim == 0 and getattr(ad, "nominal_schema", None) is None
    assert info["nominal_channel"].startswith("omitted")
    assert encoded_queries(ad, data, "test") is None
    assert ad.net[0].in_features == data.X_train.shape[1] + 3
    m = evaluate_reuse(model, ad, data.X_test, data.y_test, cfg, test_groups=data.test_groups)
    assert np.isfinite(m["accuracy_post"])
    with_du, _ = train_classification_adapter(model, data, rc("nominal_residual_scores", "combined", epochs=2,
        nominal_fields=FIELDS, nominal_fields_covered_by_representation=False))
    assert with_du.nominal_dim == 9  # the default keeps the explicit channel
    with pytest.raises(ValueError, match="uncovered"):
        rc("nominal_residual_scores", omit_nominal_channel=True).validate()
    with pytest.raises(ValueError, match="aggregate adapter only"):
        rc("per_case_2022", nominal_fields=FIELDS, nominal_fields_covered_by_representation=False,
           omit_nominal_channel=True).validate()
    covered = [dict(f, covered_by_representation=True) for f in FIELDS]  # the data documents color/shape uncovered
    with pytest.raises(ValueError, match="coverage differs"):
        train_classification_adapter(model, data, rc("nominal_residual_scores", epochs=1, nominal_fields=covered,
                                                     nominal_fields_covered_by_representation=True))


def test_default_reuse_config_is_unchanged():
    cfg = ReuseConfig("nominal_residual_scores", "combined", 1., 1., "softmax")
    assert cfg.omit_nominal_channel is False
    cfg.validate()
    with pytest.raises(ValueError):
        ReuseConfig("per_case", "combined", 1., 1., "softmax").validate()


# --------------------------------------------------------------------------- configs


def load(name):
    return yaml.safe_load((P3 / name).read_text(encoding="utf-8"))


FILES = sorted(p.name for p in P3.glob("*.yaml"))


def test_pack_files():
    assert FILES == sorted(["pretopk_pct10.yaml", "pretopk_pct25.yaml", "pretopk_pct50.yaml",
                            "pctinit_sync_classification.yaml", "pctinit_sync_regression.yaml",
                            "pvdist_sync_classification.yaml", "percase_reuse_classification.yaml",
                            "percase_nominal_reuse.yaml", "percase_reuse_regression.yaml",
                            "directhead_classification.yaml", "directhead_nominal_reuse.yaml",
                            "nodu_nominal_reuse.yaml"])


@pytest.mark.parametrize("name", FILES)
def test_every_config_declares_its_choices_and_keeps_the_rules(name):
    text = (P3 / name).read_text(encoding="utf-8")
    assert "exploratory, our choice, not a PI decision" in text.splitlines()[0]
    cfg = yaml.safe_load(text)
    assert cfg["status"] == "exploratory_handoff_20261126_p3"
    assert cfg["seeds"] == ([70, 71, 72] if name.startswith("pretopk") else [30, 31, 32])
    assert all(d["name"] != "tabarena_363697" for d in cfg["datasets"])
    assert "Study handoff_" in text and "Host" in text


@pytest.mark.parametrize("pct", [10, 25, 50])
def test_pretopk_percentile_configs_reach_the_core(pct):
    cfg = load(f"pretopk_pct{pct}.yaml")
    ref = yaml.safe_load((REPO / "configs/t1/pre_topk_expanded_20261030.yaml").read_text())
    diff = {k for k in set(cfg) | set(ref) if cfg.get(k) != ref.get(k)}
    assert diff == {"status", "core"}
    assert {k for k in cfg["core"] if cfg["core"][k] != ref["core"].get(k)} == {"bias_init", "bias_init_percentile"}
    cc = CoreConfig(task_type="classification", seed=70, pre_topk_mask=False, **cfg["core"])  # as t1_pre_topk_pilot
    assert (cc.bias_init, cc.bias_init_percentile) == ("learned_distance_percentile", pct)


@pytest.mark.parametrize("role", ["classification", "regression"])
def test_pctinit_sync_arms_reach_the_core_config(role):
    import tools.t1_run as t1_run
    cfg = load(f"pctinit_sync_{role}.yaml")
    assert [a["name"] for a in cfg["arms"]] == ["pct10", "pct25", "pct50"]
    assert [c.get("name", c.get("schedule")) for c in cfg["conditions"]] == ["independent", "alternating_rr",
                                                                            "rr_initial_bias_radius"]
    assert cfg["core"]["bias_init"] == "knn_mean"  # the arms change it, the base stays the recorded one
    for cond in cfg["conditions"]:
        t1_run.sync_maintenance_for(cond, cfg, 30)
    runner = t1_run.Runner.__new__(t1_run.Runner)
    runner.cfg = cfg
    for ds in cfg["datasets"]:
        data = T1Data(name=ds["name"], task_type=ds["task_type"], X_train=None, y_train=None, X_val=None, y_val=None,
                      X_test=None, y_test=None)
        for arm in cfg["arms"]:
            cc = runner.core_config(data, 30, **t1_run.arm_core(arm))
            assert (cc.bias_init, cc.bias_init_percentile) == ("learned_distance_percentile", arm["bias_init_percentile"])
            tuned = cfg["core_by_dataset"].get(f"{ds['task_type']}__{ds['name']}", {})  # synthetic_diag: base core
            assert all(getattr(cc, k) == (tuple(v) if isinstance(v, list) else v) for k, v in tuned.items())


def test_pvdist_conditions_reach_the_calibration(monkeypatch):
    import tools.t1_run as t1_run
    cfg = load("pvdist_sync_classification.yaml")
    assert all(c["free_radius_label_metric"] == "probability_vector_distance" for c in cfg["conditions"])
    for cond in cfg["conditions"]:
        t1_run.sync_maintenance_for(cond, cfg, 30)  # an unread key would raise here
    src = Path(t1_run.__file__).read_text()
    assert 'label_metric=merged.get("free_radius_label_metric")' in src


@pytest.mark.parametrize("name,modes", [("percase_reuse_classification.yaml", {"per_case_2022"}),
                                        ("percase_nominal_reuse.yaml", {"per_case_2022"}),
                                        ("percase_reuse_regression.yaml", {"per_case_2022"}),
                                        ("directhead_classification.yaml", {"direct_head"}),
                                        ("directhead_nominal_reuse.yaml", {"direct_head"}),
                                        ("nodu_nominal_reuse.yaml", {"nominal_residual_scores", "logit_residual"})])
def test_reuse_conditions_reach_reuse_config(name, modes):
    import tools.t1_run as t1_run
    cfg = load(name)
    assert cfg["type"] == "reuse"
    assert {c["output_mode"] for c in cfg["conditions"]} == modes
    runner = t1_run.Runner.__new__(t1_run.Runner)
    runner.cfg = cfg
    labels = []
    for cond in cfg["conditions"]:
        r = ReuseConfig(seed=30, **{**cfg["reuse"], **{k: v for k, v in cond.items() if k != "name"}})  # as reuse_condition
        r.validate()
        assert r.output_mode == cond["output_mode"]
        assert r.omit_nominal_channel is (name == "nodu_nominal_reuse.yaml")
        labels.append(runner.reuse_unit(cfg["datasets"][0]["task_type"], cond)[0])
    assert len(set(labels)) == len(labels)
    if name == "percase_reuse_regression.yaml":
        assert labels == ["per_case_2022_regression_diff_only", "per_case_2022_regression_combined"]
