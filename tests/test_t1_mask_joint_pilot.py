"""Check that the factorial pilot isolates retrieval trainability after matched warm-up."""
import json
import shutil

import pytest
import torch
from model.t1 import records
from model.t1.data import T1Data
import tools.t1_mask_joint_pilot as pilot


@pytest.fixture(autouse=True)
def preserve_torch_process_settings():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng = torch.get_rng_state()
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.set_rng_state(rng)


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
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    rows = pilot.run(study, tmp_path / "pilot", "configs/t1/mask_joint_pilot_20261030.yaml", "mj")
    assert len(rows) == 4
    def timeless(obj):  # wall times aside
        if isinstance(obj, dict):
            return {k: timeless(v) for k, v in obj.items() if "seconds" not in k}
        return [timeless(v) for v in obj] if isinstance(obj, list) else obj
    rows = [records.jsonable(r) for r in rows]
    # One continuation lost: the rerun loads its mask setting's pretrained core (never pretrained again), continues
    # only the lost one and reads the others back; every row equals the first run's.
    name = rows[1]["dataset"]
    shutil.rmtree(tmp_path / "runs" / "t1" / "mj" / name / "s3" / "mask1_frozen0")
    with monkeypatch.context() as m:
        m.setattr(pilot, "train_retrieval", lambda *a, **k: pytest.fail("pretrained again"))
        again = pilot.run(study, tmp_path / "pilot2", "configs/t1/mask_joint_pilot_20261030.yaml", "mj")
    assert json.dumps(timeless([records.jsonable(r) for r in again])) == json.dumps(timeless(rows))
    # A rerun of the complete study reads every continuation back: nothing is pretrained or continued again.
    monkeypatch.setattr(pilot, "train_retrieval", lambda *a, **k: pytest.fail("pretrained again"))
    monkeypatch.setattr(pilot, "continuation", lambda *a, **k: pytest.fail("continued again"))
    monkeypatch.setattr(pilot.records, "stage_unit", lambda *a, **k: pytest.fail("pretrain loaded without need"))
    again = pilot.run(study, tmp_path / "pilot3", "configs/t1/mask_joint_pilot_20261030.yaml", "mj")
    assert json.dumps(timeless([records.jsonable(r) for r in again])) == json.dumps(timeless(rows))
    for masked in [True, False]:
        frozen, joint = [next(r for r in rows if r["masked"] == masked and r["frozen"] == f) for f in [True, False]]
        assert frozen["warmup_end_sha256"] == joint["warmup_end_sha256"]
        assert frozen["pretrained_state_sha256"] == frozen["last_retriever_sha256"]
        assert joint["last_retriever_sha256"] != joint["pretrained_state_sha256"]
        assert frozen["selected"]["pre_prediction"] == frozen["baseline"]["pre_prediction"]
        assert frozen["last"]["pre_loss"] == frozen["baseline"]["pre_loss"]
