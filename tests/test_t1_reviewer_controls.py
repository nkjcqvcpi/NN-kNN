import copy

import numpy as np
import pytest
import torch

from model.t1.core import CoreConfig, build_model, make_optimizer, train_retrieval
from model.t1.maintenance import CaseArchive, run_maintenance
from model.t1.provenance import CaseStatisticsStore, ScoreConfig
from model.t1.retention import RetentionConfig
from model.t1.revise import InterventionLog


def toy(task="classification"):
    X = torch.tensor([[0., 0.], [.1, .2], [.2, .1], [1., 1.], [1.1, 1.], [1., 1.2]])
    y = torch.tensor([0, 0, 0, 1, 1, 1]) if task == "classification" else torch.arange(6.).float()
    cc = CoreConfig(task_type=task, top_k=2, epochs=2, batch_size=3)
    model = build_model(X, y, cc, 2 if task == "classification" else None)
    model.eval()
    return X, y, model, cc


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_force_is_transient_uses_actual_neighborhood_and_floor(task):
    X, _, model, _ = toy(task)
    with torch.no_grad():
        model.biases[5] = -10000
    baseline = model.retrieve(X[:1], exclude_identical=False)
    assert baseline["weights"][0, 5] == 0
    original = copy.deepcopy(model.state_dict())
    log = InterventionLog("forced")
    pre, final, iid = log.controlled_decision(model, X[:1], [5], query_ids=["q0"],
                                               activation_floor=.1, actor="reviewer", reason="controlled contrast")
    event = log.decisions[0]
    assert event["intervention_id"] == iid
    assert len(event["case_ids"]) <= 2
    assert event["activations"][event["case_ids"].index(5)] >= .1
    labels = model.labels[torch.tensor(event["case_ids"])].view(len(event["case_ids"]), -1)
    assert torch.allclose(pre, torch.tensor(event["activations"]).unsqueeze(0) @ labels)
    assert torch.equal(pre, final)
    assert sum(event["activations"]) == pytest.approx(1.)
    assert torch.equal(model.retrieve(X[:1], exclude_identical=False)["weights"], baseline["weights"])
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())


@pytest.mark.parametrize("ids,floor", [([99], .1), ([1, 1], .1), ([1, 2, 3], .1), ([1], None), ([1], 0), ([1], float("nan")), ([1, 2], .6), ([1.5], .1)])
def test_force_invalid_requests_fail(ids, floor):
    X, _, model, _ = toy()
    with pytest.raises(ValueError):
        model.retrieve(X[:1], exclude_identical=False, force_case_ids=ids, forced_weight_floor=floor)


def test_force_respects_masks_self_identity_and_stable_ids_after_compaction():
    X, _, model, _ = toy()
    mask = torch.ones(6, dtype=torch.bool)
    mask[5] = False
    for kwargs in ({"case_mask": mask}, {"query_case_ids": [5]}):
        with pytest.raises(ValueError, match="bypass"):
            model.retrieve(X[:1], exclude_identical=False, force_case_ids=[5], forced_weight_floor=.1, **kwargs)
    model.compact_cases(torch.tensor([0, 2, 5]))
    r = model.retrieve(X[:1], exclude_identical=False, force_case_ids=[5], forced_weight_floor=.1)
    assert r["weights"][0, 2] >= .1
    with pytest.raises(ValueError, match="active"):
        model.retrieve(X[:1], exclude_identical=False, force_case_ids=[1], forced_weight_floor=.1)
    model.config["pre_topk_mask"] = False
    with pytest.raises(ValueError, match="pre-top-k"):
        model.retrieve(X[:1], force_case_ids=[5], forced_weight_floor=.1)


def test_parameter_edits_are_validated_versioned_and_undoable():
    _, _, model, _ = toy()
    log = InterventionLog("edits")
    before = model.glocal_weightor.feature_weights.clone()
    iid = log.set_feature_weights(model, [[3., .01]], actor="reviewer", reason="known redundant feature")
    log.undo_parameter_edit(model, iid, actor="reviewer", reason="rollback")
    assert torch.equal(before, model.glocal_weightor.feature_weights)
    assert log.records[-1]["undo_of"] == iid
    assert [r["version"] for r in log.records] == [1, 2]
    bias = model.biases[1].clone()
    iid = log.adjust_bias(model, 1, .5, actor="reviewer", reason="test")
    log.undo_parameter_edit(model, iid, actor="reviewer", reason="rollback")
    assert torch.equal(model.biases[1], bias)
    for value in ([[float("nan"), 1.]], [[1.]]):
        with pytest.raises(ValueError):
            log.set_feature_weights(model, value, actor="reviewer", reason="invalid")
    with pytest.raises(ValueError):
        log.adjust_bias(model, 1, float("inf"), actor="reviewer", reason="invalid")
    iid = log.set_feature_weights(model, [[2., 1.]], actor="reviewer", reason="first")
    log.set_feature_weights(model, [[3., 1.]], actor="reviewer", reason="later")
    with pytest.raises(ValueError, match="stale"):
        log.undo_parameter_edit(model, iid, actor="reviewer", reason="unsafe stale rollback")


def test_human_protection_survives_maintenance_and_restore_preserves_quarantine():
    X, y, model, cc = toy()
    optimizer = train_retrieval(model, X, y, X + .02, y, cc).optimizer
    store = CaseStatisticsStore()
    log = InterventionLog("protect")
    log.protect(model, 5, store, actor="reviewer", reason="designated case")
    with pytest.raises(ValueError, match="Unprotect"):
        log.archive_remove(model, 5, store, actor="reviewer", reason="invalid", optimizer=optimizer)
    archive = CaseArchive()
    sc = ScoreConfig(smoothing=1., min_retrieval_count=1, min_activation_mass=.1)
    result = run_maintenance(model, store, archive, 1, RetentionConfig("random", 0), sc,
                             step=1, run_id="protected", optimizer=optimizer)
    assert model.active_case_ids().tolist() == [5]
    assert "human_protect" in next(e for e in result["events"] if e["case_id"] == 5)["reason_codes"]
    _, _, model, _ = toy()
    log = InterventionLog("restore")
    optimizer = make_optimizer(model, cc)
    train_retrieval(model, X, y, X + .02, y, cc, optimizer=optimizer)
    before = {n: getattr(model, n)[2].detach().clone() for n in ("cases", "labels", "biases", "glocal_weights", "case_ids")}
    log.quarantine(model, 2, actor="reviewer", reason="hold")
    log.archive_remove(model, 2, store, actor="reviewer", reason="temporary", optimizer=optimizer)
    log = InterventionLog.from_state_dict(log.state_dict())
    log.restore(model, 2, actor="reviewer", reason="restore", optimizer=optimizer)
    slot = model.active_case_ids().tolist().index(2)
    assert all(torch.equal(v, getattr(model, n)[slot]) for n, v in before.items())
    assert not log.case_mask(model)[slot]
    assert optimizer.state[model.biases]["exp_avg"][slot] == 0
    log.release(model, 2, actor="reviewer", reason="release")
    assert log.case_mask(model) is None


def test_archive_restore_invalid_batch_is_atomic():
    _, _, model, _ = toy()
    archive = CaseArchive()
    archive.add(model, 4, {}, 0, ["test"])
    archive.add(model, 5, {}, 0, ["test"])
    model.compact_cases(torch.arange(5))  # ID4 still active, ID5 can restore
    state = copy.deepcopy(model.state_dict())
    for ids in ([5, 999], [5, 5], [4], [5, 4]):
        with pytest.raises((ValueError, RuntimeError)):
            archive.restore(model, ids)
        assert model.case_count() == 5
        assert len(archive.entries) == 2
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
