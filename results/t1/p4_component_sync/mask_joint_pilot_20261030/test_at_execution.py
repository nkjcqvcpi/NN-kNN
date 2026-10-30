"""Check that the factorial pilot isolates retrieval trainability after matched warm-up."""
import pytest
import torch
from model.t1.data import T1Data
import tools.t1_mask_joint_pilot as pilot


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_paired_freezing_preserves_retriever_and_matches_warmup(tmp_path, monkeypatch, task):
    study = dict(seeds=[3], splits=dict(test_frac=.2, val_frac=.2),
                 core=dict(epochs=3, patience=3, batch_size=8, top_k=3, bias_init_k=3),
                 sync=dict(schedule="joint_rr", joint_objective="summed", w_pre=1., w_post=.5, w_near=0.,
                           v_post=1., v_delta=1., v_small=.1, hidden_dims=[8,4], epochs=4, patience=4,
                           adapter_warmup_epochs=1, diagnostics_every=0),
                 datasets=[dict(name="linear_regression", task_type="regression", generator=dict(n=40,d=5,noise_scale=.5))])
    if task == "classification":
        X = torch.randn(40, 3, generator=torch.Generator().manual_seed(8))
        y = (X[:,0] > 0).long()
        data = T1Data("tiny", task, X[:24], y[:24], X[24:32], y[24:32], X[32:], y[32:], num_classes=2)
        monkeypatch.setattr(pilot, "data_for", lambda *a, **kw: data)
        study["datasets"] = [dict(name="tiny", task_type=task)]
    rows = pilot.run(study, tmp_path / "pilot", "configs/t1/mask_joint_pilot_20261030.yaml")
    assert len(rows) == 4
    for masked in [True, False]:
        frozen, joint = [next(r for r in rows if r["masked"] == masked and r["frozen"] == f) for f in [True, False]]
        assert frozen["warmup_end_sha256"] == joint["warmup_end_sha256"]
        assert frozen["pretrained_state_sha256"] == frozen["last_retriever_sha256"]
        assert joint["last_retriever_sha256"] != joint["pretrained_state_sha256"]
        assert frozen["selected"]["pre_prediction"] == frozen["baseline"]["pre_prediction"]
        assert frozen["last"]["pre_loss"] == frozen["baseline"]["pre_loss"]
