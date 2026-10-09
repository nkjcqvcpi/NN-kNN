"""Causal revision artifacts must replay the predictor actually measured."""
import copy
from types import SimpleNamespace

import pytest
import torch

from model.common.core import CoreConfig, build_model, evaluate, make_optimizer, train_retrieval
from model.t1.revise import InterventionLog, evaluate_m0_m1_m2


@pytest.mark.parametrize("operation", ["relabel", "quarantine"])
@pytest.mark.parametrize("epochs", [0, 2])
def test_stage_artifacts_replay_and_preserve_original(tmp_path, operation, epochs):
    X = torch.tensor([[0.], [.1], [.2], [.8], [.9], [1.]])
    y = torch.tensor([0, 1, 0, 1, 1, 1])
    cc = CoreConfig(task_type="classification", epochs=2, top_k=2, batch_size=3)
    model = build_model(X, y, cc, 2)
    tr = train_retrieval(model, X, y, X + .02, y, cc)
    original = copy.deepcopy(model.state_dict())
    data = SimpleNamespace(X_train=X, y_train=y, X_val=X + .02, y_val=y,
                           X_test=X + .03, y_test=y)
    corrected = y.clone()
    corrected[1] = 0
    log = InterventionLog("test")
    def edit(m, lg):
        if operation == "relabel":
            lg.relabel(m, 1, 0, actor="test", reason="known corrupted label")
        else:
            lg.quarantine(m, 1, actor="test", reason="known corrupted label")
        return [1]
    result = evaluate_m0_m1_m2(model, data, cc, edit, log, retrain_epochs=epochs,
                             y_train=corrected, core_optimizer=tr.optimizer,
                             capture_artifacts=True)
    artifacts = result["_artifacts"]
    for stage, checkpoint in artifacts["stages"].items():
        path = tmp_path / f"{stage}.pt"
        torch.save(checkpoint, path)
        saved = torch.load(path, weights_only=False)
        replay = copy.deepcopy(model)
        if stage == "M2" and operation == "quarantine":
            replay.compact_cases(torch.tensor([0, 2, 3, 4, 5]))
        replay.load_state_dict(saved["model_state"])
        for n, p in replay.named_parameters():
            p.requires_grad_(saved["requires_grad"][n])
        metrics = evaluate(replay, data.X_test, y, case_mask=saved["case_mask"])
        assert metrics["loss_pre"] == pytest.approx(result[stage]["loss_pre"], abs=1e-7)
        assert metrics["accuracy_pre"] == result[stage]["accuracy_pre"]
        optimizer = make_optimizer(replay, cc)
        optimizer.load_state_dict(saved["optimizer_state"])
        # Aligned moments support the next real optimization step after removal.
        if stage == "M2":
            train_retrieval(replay, X, corrected, data.X_val, y, cc,
                            epochs=1, optimizer=optimizer, select_best=False)
    assert torch.equal(artifacts["y_train_corrected"], corrected)
    assert len(artifacts["history"]) == epochs
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    assert torch.equal(y, torch.tensor([0, 1, 0, 1, 1, 1]))


def test_invalid_relabel_is_atomic():
    X = torch.tensor([[0.], [1.]])
    model = build_model(X, torch.tensor([0, 1]), CoreConfig(task_type="classification"), 2)
    original = model.labels.clone()
    log = InterventionLog("test")
    for label in (-1, 2, .5):
        with pytest.raises(ValueError):
            log.relabel(model, 0, label, actor="test", reason="invalid")
        assert torch.equal(model.labels, original)
        assert not log.records


def test_matched_control_no_edit_is_identical_with_dropout_and_common_rng():
    X = torch.tensor([[0.], [.1], [.2], [.8], [.9], [1.]])
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    cc = CoreConfig(task_type="classification", representation="mlp", mlp_dims=(4,),
                    mlp_dropout=.25, top_k=2, epochs=2, batch_size=3)
    model = build_model(X, y, cc, 2)
    tr = train_retrieval(model, X, y, X + .01, y, cc)
    data = SimpleNamespace(X_train=X, y_train=y, X_val=X + .01, y_val=y,
                           X_test=X + .02, y_test=y)
    rng = torch.random.get_rng_state().clone()
    result = evaluate_m0_m1_m2(model, data, cc, lambda m, lg: [], InterventionLog("matched"),
                              retrain_epochs=2, core_optimizer=tr.optimizer,
                              capture_artifacts=True, matched_training_control=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    stages = result["_artifacts"]["stages"]
    assert result["MC"] == result["M2"]
    assert result["M2_minus_MC_loss"] == 0
    assert all(torch.equal(v, stages["MC"]["model_state"][k]) for k, v in stages["M2"]["model_state"].items())
    assert len(result["_artifacts"]["control_history"]) == 2


def test_no_training_omits_conditional_control():
    X = torch.tensor([[0.], [1.]])
    y = torch.tensor([0, 1])
    cc = CoreConfig(task_type="classification")
    model = build_model(X, y, cc, 2)
    data = SimpleNamespace(X_train=X, y_train=y, X_val=X, y_val=y, X_test=X, y_test=y)
    result = evaluate_m0_m1_m2(model, data, cc, lambda m, lg: [], InterventionLog("none"),
                              retrain_epochs=0, matched_training_control=True)
    assert "MC" not in result
