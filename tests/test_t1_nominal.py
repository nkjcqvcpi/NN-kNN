"""Actual nominal reuse paths, train-only vocabulary and stable-ID events."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from model.nn_cdh import ClassificationNNCDHAdapter
from model.common.core import CoreConfig, build_model, retrieval_events
from model.common.nominal import NominalSchema, NominalClassificationAdapter, encoded_queries
from model.common.outcomes import final_prediction, prediction_loss
from model.t1.reuse import ReuseConfig, train_classification_adapter, evaluate_reuse, neighborhood_inputs
from model.t1.provenance import CaseStatisticsStore, audit_provenance
from model.t1.candidates import MaintenanceReference, removal_influence, coverage_reachability
from model.t1.data import make_nominal_synthetic
from model.t1.sync import SyncConfig, train_synchronized

FIELDS = [{"name": "color", "covered_by_representation": False},
          {"name": "shape", "covered_by_representation": False},
          {"name": "size", "covered_by_representation": True}]


def fixture():
    X = torch.tensor([[0., 0.], [.1, .2], [.3, .4], [1., .9], [1.1, .8], [1.3, .6]])
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    values = {"color": ["red", "red", "blue", "blue", "green", None],
              "shape": ["circle", "square"] * 3, "size": ["small"] * 3 + ["large"] * 3}
    schema = NominalSchema(FIELDS).fit(values)
    model = build_model(X, y, CoreConfig(task_type="classification", top_k=2), 3)
    ids = torch.tensor([11, 5, 42, 8, 19, 3])
    model.case_ids[:6].copy_(ids)
    model.next_case_id.fill_(43)
    ad = NominalClassificationAdapter(schema=schema, case_ids=ids, case_values=schema.encode(values),
        feature_dim=2, num_classes=3, hidden_dims=(8, 4))
    query = schema.encode({"color": ["red", "amber"], "shape": ["square", "circle"], "size": ["small", "large"]})
    return model, ad, X[:2] + .03, query


def test_training_only_vocab_unknown_missing_and_covered_field_groups():
    schema = NominalSchema(FIELDS).fit({"color": ["red", "__UNKNOWN__", None],
        "shape": [1, "1", True], "size": ["small"] * 3})
    snapshot = schema.state_dict()
    assert [g["name"] for g in snapshot["groups"]] == ["color", "shape"]
    assert len(snapshot["groups"][1]["categories"]) == 3  # typed categories stay distinct
    q = schema.encode({"color": ["amber", None, "__UNKNOWN__"],
        "shape": [1, "never-seen", None], "size": ["large"] * 3})
    assert q[0, 1] == q[1, 0] == 1
    assert q[2, :snapshot["groups"][0]["width"]].argmax() >= 2
    for group in snapshot["groups"]:
        assert torch.equal(q[:, group["offset"]:group["offset"] + group["width"]].sum(1), torch.ones(3))
    assert schema.state_dict() == snapshot
    snapshot["groups"][0]["categories"].append("mutated exported metadata")
    assert schema.state_dict() != snapshot
    with pytest.raises(ValueError, match="refit"):
        schema.fit({"color": ["amber"], "shape": ["circle"], "size": ["large"]})


@pytest.mark.parametrize("fields", [[{"name": "x"}], [{"name": "x", "covered_by_representation": 1}],
    [{"name": "x", "covered_by_representation": False}] * 2])
def test_invalid_field_coverage_is_not_inferred(fields):
    with pytest.raises(ValueError):
        NominalSchema(fields)


def test_nominal_dimensions_and_missing_inputs_fail_explicitly():
    ad = ClassificationNNCDHAdapter(2, 3, nominal_dim=4, hidden_dims=(8, 4))
    with pytest.raises(ValueError, match="actual query"):
        ad(torch.zeros(2, 2), torch.ones(2, 3) / 3)
    covered = ClassificationNNCDHAdapter(2, 3, nominal_dim=0, hidden_dims=(8, 4))
    with pytest.raises(ValueError, match="duplicated"):
        covered(torch.zeros(2, 2), torch.ones(2, 3) / 3, torch.zeros(2, 4))
    cfg = ReuseConfig("nominal_residual_scores", "combined", 1., 1., "softmax")
    with pytest.raises(ValueError, match="field-by-field"):
        replace(cfg, nominal_fields_covered_by_representation=False).validate()
    with pytest.raises(ValueError, match="agree"):
        replace(cfg, nominal_fields=FIELDS).validate()


def test_grouped_delta_uses_actual_weights_and_survives_slot_compaction():
    model, ad, X, query = fixture()
    keep = torch.tensor([4, 1, 5, 2])
    mask = torch.zeros(6, dtype=torch.bool)
    mask[keep] = True
    r = model.retrieve(X, case_mask=mask, exclude_identical=False)
    expected = query - r["weights"] @ ad.values_for_ids(model.active_case_ids())
    before = ad.nominal_delta(model, r, query)
    torch.testing.assert_close(before, expected)
    model.compact_cases(keep)
    r2 = model.retrieve(X, exclude_identical=False)
    after = ad.nominal_delta(model, r2, query)
    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)
    for group in ad.nominal_schema.groups:
        assert torch.allclose(after[:, group["offset"]:group["offset"] + group["width"]].sum(1), torch.zeros(2), atol=1e-6)


def test_nominal_queries_require_real_groups_and_known_stable_bindings():
    model, ad, X, query = fixture()
    r = model.retrieve(X, exclude_identical=False)
    for invalid in (None, query[:, :-1], query * .5, query * float("nan")):
        with pytest.raises(ValueError):
            ad.nominal_delta(model, r, invalid)
    model.case_ids[0] = 999
    with pytest.raises(ValueError, match="binding"):
        ad.nominal_delta(model, r, query)


def test_forward_and_outcome_share_event_with_no_raw_embedding_leakage():
    model, ad, X, query = fixture()
    model.classification_adapter = ad
    model._current_Delta_u_q = torch.zeros_like(query)  # stale private context cannot bypass explicit input
    with pytest.raises(ValueError, match="explicit encoded"):
        model(X)
    captured = []
    handle = ad.net.register_forward_pre_hook(lambda m, inputs: captured.append(inputs[0].detach().clone()))
    r = model.retrieve(X, exclude_identical=False)
    pre, post, _ = final_prediction(model, X, r, query_nominal=query)
    expected = torch.cat([r["query_features"] - r["weights"] @ r["case_features"],
        ad.nominal_delta(model, r, query), pre], 1)
    torch.testing.assert_close(captured[-1], expected)
    torch.testing.assert_close(model(X, query_nominal=query)[0], post)
    handle.remove()
    model.enable_classification_adapter = False
    torch.testing.assert_close(model(X)[0], pre)


def test_nominal_checkpoint_preserves_vocab_and_predictions(tmp_path):
    model, ad, X, query = fixture()
    p = tmp_path / "nominal.pt"
    torch.save({"state": ad.state_dict(), "schema": ad.nominal_schema.state_dict()}, p)
    saved = torch.load(p, weights_only=True)
    restored = NominalClassificationAdapter(schema=NominalSchema.from_state_dict(saved["schema"]),
        case_ids=saved["state"]["nominal_case_ids"], case_values=saved["state"]["nominal_case_values"],
        feature_dim=2, num_classes=3, hidden_dims=(8, 4))
    restored.load_state_dict(saved["state"])
    r = model.retrieve(X, exclude_identical=False)
    a = final_prediction(model, X, r, adapter=ad, query_nominal=query)[1]
    b = final_prediction(model, X, r, adapter=restored, query_nominal=query)[1]
    assert torch.equal(a, b)
    assert restored.nominal_schema.state_dict() == ad.nominal_schema.state_dict()


def test_nominal_final_outcome_credit_and_counterfactual_candidates():
    model, ad, X, query = fixture()
    y = torch.tensor([0, 1])
    r = model.retrieve(X, exclude_identical=False)
    p = final_prediction(model, X, r, adapter=ad, query_nominal=query)[1]
    store = CaseStatisticsStore()
    info = audit_provenance(model, X, y, store, exclude_identical=False, counterfactual=False,
        adapter=ad, query_nominal=query)
    assert info["mean_loss_final"] == pytest.approx(float(prediction_loss(p, y, "cross_entropy").mean().detach()))
    assert sum(s.activation_mass for s in store._stats.values()) == pytest.approx(2)
    assert all(s.positive_support+s.harmful_support == pytest.approx(s.activation_mass) for s in store._stats.values())
    ref = MaintenanceReference(X, y, adapter=ad, stream="maintenance_split", query_nominal=query)
    full = removal_influence(model, ref)
    cached = removal_influence(model, ref, cached_threshold=0.)
    np.testing.assert_allclose(full["full"], cached["cached"], atol=1e-6)
    ratio = coverage_reachability(model, ref, activation_threshold=0., zero_reachability="zero")
    assert len(ratio["reachability"]) == model.case_count()
    events = retrieval_events(model, X, y, stream="test", run_id="nominal", adapter=ad, query_nominal=query)
    assert [e["final_prediction"] for e in events] == p.argmax(1).tolist()
    du = ad.nominal_delta(model, r, query)
    for i, event in enumerate(events):
        torch.testing.assert_close(torch.tensor(event["delta_u"]), du[i])
        torch.testing.assert_close(torch.tensor(event["p0"]) + torch.tensor(event["predicted_residual"]), p[i])
        torch.testing.assert_close(torch.tensor(event["p0"]) + torch.tensor(event["residual_target"]),
                                   torch.nn.functional.one_hot(y[i], 3).float())
    with pytest.raises(ValueError, match="stable case-ID binding"):
        neighborhood_inputs(model, X, exclude_identical=False, X_nominal=query,
                            nominal_case=ad.nominal_case_values)


def test_nominal_training_fits_only_train_and_keeps_retrieval_frozen():
    data = make_nominal_synthetic(8)
    model = build_model(data.X_train, data.y_train, CoreConfig(task_type="classification"), 3)
    before = copy.deepcopy(model.state_dict())
    cfg = ReuseConfig("nominal_residual_scores", "combined", 1., 1., "softmax", epochs=2,
        hidden_dims=(8, 4), nominal_fields=FIELDS, nominal_fields_covered_by_representation=False)
    ad, info = train_classification_adapter(model, data, cfg)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert ad.nominal_dim == 9
    assert not any("amber" in token for g in info["nominal_manifest"]["groups"] for token in g["categories"])
    metrics = evaluate_reuse(model, ad, data.X_test, data.y_test, cfg,
        query_nominal=encoded_queries(ad, data, "test"), test_groups=data.test_groups)
    assert metrics["n_known"] == 63 and metrics["n_unknown"] == 18 and metrics["n_missing"] == 9
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))


def test_nominal_synchronization_uses_grouped_inputs():
    data = make_nominal_synthetic(9)
    cc = CoreConfig(task_type="classification", batch_size=90)
    model = build_model(data.X_train, data.y_train, cc, 3)
    cfg = SyncConfig("alternating_rr", 1., .5, .1, 1., 1., .1, epochs=4,
        block_epochs=2, hidden_dims=(8, 4), nominal_fields=FIELDS)
    ad, info = train_synchronized(model, data, cc, cfg, None, near_scale=1.)
    assert ad.nominal_dim == 9 and info["best_epoch"] >= 3
    assert all(np.isfinite(h["val_L_post"]) for h in info["history"])
    assert len(encoded_queries(ad, data, "test")) == len(data.y_test)


def test_nominal_rrr_maintains_original_bindings_after_optimizer_compaction():
    from model.t1.maintenance import realign_optimizer_state
    data = make_nominal_synthetic(10)
    cc = CoreConfig(task_type="classification", batch_size=90)
    model = build_model(data.X_train, data.y_train, cc, 3)
    cfg = SyncConfig("alternating_rrr", 1., .5, .1, 1., 1., .1, epochs=6,
        block_epochs=2, hidden_dims=(8, 4), nominal_fields=FIELDS)
    def hook(ep, mdl, opt, ad):
        before = mdl.case_count()
        keep = np.arange(0, before, 2)
        mdl.compact_cases(keep)
        realign_optimizer_state(opt, mdl, keep, before)
        assert torch.equal(ad.values_for_ids(mdl.active_case_ids()),
                           ad.nominal_schema.encode(data.nominal_train)[mdl.active_case_ids()])
        return {"n_before": before, "n_after": mdl.case_count()}
    ad, info = train_synchronized(model, data, cc, cfg, None, near_scale=1.,
                                 maintenance_hook=hook, maintenance_epochs={4})
    assert model.case_count() == 90 and info["best_epoch"] >= 4
    assert info["history"][3]["maintenance_state_changed"]
    query = encoded_queries(ad, data, "test")
    r = model.retrieve(data.X_test, exclude_identical=False)
    expected = query - r["weights"] @ ad.nominal_schema.encode(data.nominal_train)[model.active_case_ids()[r["case_indices"]]]
    torch.testing.assert_close(ad.nominal_delta(model, r, query), expected)
