"""Joint (end-to-end) synchronization and the EMA/encoder ablation runner."""
import copy
import json
from pathlib import Path

import pytest
import torch

from model.common.core import CoreConfig, build_model, train_retrieval
from model.t1.calibration import calibrate_free_radius
from model.t1.data import T1Data
from model.t1.sync import SyncConfig, _forward_terms, joint_backward, train_synchronized
from model.nn_cdh import ClassificationNNCDHAdapter
from tools.t1_run import EveryEpoch, Runner, load_config


def classification_data(n=24):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, 3, generator=g)
    y = (X[:, 0] > 0).long()
    return T1Data(name="tiny", task_type="classification", X_train=X, y_train=y,
                  X_val=X[:8] + .05, y_val=y[:8], X_test=X[8:16] - .05, y_test=y[8:16], num_classes=2)


def regression_data():
    x = torch.tensor([[0.], [.1], [.3], [.8], [.9], [1.2]])
    y = torch.tensor([0., .1, .2, .8, 1., 1.2])
    return T1Data(name="tiny", task_type="regression", X_train=x, y_train=y,
                  X_val=x + .01, y_val=y, X_test=x + .02, y_test=y)


def sync_cfg(schedule, task="classification", **kw):
    return SyncConfig(schedule, 1., .5, .1, 1., 1., .1,
                      output_mode="aggregate_regression" if task == "regression" else "nominal_residual_scores",
                      epochs=kw.pop("epochs", 4), patience=4, block_epochs=1, hidden_dims=(8, 4), **kw)


@pytest.mark.parametrize("representation,mcb", [("raw", False), ("mlp", False), ("mlp", True)])
def test_separated_joint_backward_matches_component_objectives(representation, mcb):
    data = classification_data()
    cc = CoreConfig(task_type="classification", top_k=4, representation=representation, mcb_enabled=mcb)
    model = build_model(data.X_train, data.y_train, cc, 2)
    fr, _ = calibrate_free_radius(model, s_task=.5, snapshot_step=0)
    adapter = ClassificationNNCDHAdapter(model.retrieve(data.X_train[:1], exclude_identical=False)["query_features"].shape[1],
                                         2, 0, (8, 4), "nominal_residual_scores")
    cfg = sync_cfg("joint_rr")
    rp = [p for p in model.parameters() if p.requires_grad]
    ap = [p for p in adapter.parameters() if p.requires_grad]
    ids = torch.arange(8)

    def terms():
        return _forward_terms(model, adapter, data.X_train[:8], data.y_train[:8], fr, output_mode=cfg.output_mode,
                              near_scale=1., query_case_ids=ids)

    T = terms()
    L_R = cfg.w_pre * T["L_pre"] + cfg.w_post * T["L_post"] + cfg.w_near * T["L_near"]
    L_A = cfg.v_post * T["L_post"] + cfg.v_delta * T["L_delta"] + cfg.v_small * T["L_small"]
    expected_r = torch.autograd.grad(L_R, rp, retain_graph=True, allow_unused=True)
    expected_a = torch.autograd.grad(L_A, ap, allow_unused=True)
    joint_backward(terms(), cfg, rp, ap)
    for p, g in zip(rp, expected_r):
        assert torch.allclose(p.grad if p.grad is not None else torch.zeros_like(p),
                              g if g is not None else torch.zeros_like(p), atol=1e-6)
    for p, g in zip(ap, expected_a):
        assert torch.allclose(p.grad if p.grad is not None else torch.zeros_like(p),
                              g if g is not None else torch.zeros_like(p), atol=1e-6)
    # L_post reaches retrieval through the adapter: retrieval gradients differ
    # from those of the retrieval-only objective.
    pre_only = torch.autograd.grad(terms()["L_pre"], rp, allow_unused=True)
    assert any(g is not None and not torch.allclose(p.grad, g) for p, g in zip(rp, pre_only))
    for p in rp + ap:
        p.grad = None
    joint_backward(terms(), sync_cfg("joint_rr", joint_objective="summed"), rp, ap)
    summed_r = [p.grad for p in rp]
    assert any(s is not None and e is not None and not torch.allclose(s, e) for s, e in zip(summed_r, expected_r))


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_joint_updates_both_components_every_epoch(task):
    data = classification_data() if task == "classification" else regression_data()
    cc = CoreConfig(task_type=task, batch_size=4, top_k=3, bias_init_k=3)
    model = build_model(data.X_train, data.y_train, cc, data.num_classes)
    fr, _ = calibrate_free_radius(model, s_task=.5, snapshot_step=0)
    flags = [p.requires_grad for p in model.parameters()]
    adapter, info = train_synchronized(model, data, cc, sync_cfg("joint_rr", task, adapter_warmup_epochs=1), fr, near_scale=1.)
    assert [h["phase"] for h in info["history"]][:2] == ["adapter", "joint"]
    assert [p.requires_grad for p in model.parameters()] == flags
    joint = info["history"][1]
    # A joint epoch changes retrieval terms (unlike an adapter-only epoch).
    assert joint["train_after_update"]["L_pre"] != pytest.approx(joint["train_before_update"]["L_pre"], abs=1e-10)
    assert joint["train_after_update"]["L_post"] != pytest.approx(joint["train_before_update"]["L_post"], abs=1e-10)
    warmup = info["history"][0]
    assert warmup["train_after_update"]["L_pre"] == pytest.approx(warmup["train_before_update"]["L_pre"], abs=1e-8)
    assert info["best_epoch"] >= 1 and info["best_val_L_post"] == pytest.approx(
        min(h["val_L_post"] for h in info["history"]), abs=1e-9)


def test_only_rrr_joint_schedule_maintains():
    data = regression_data()
    cc = CoreConfig(task_type="regression", batch_size=3, top_k=3, bias_init_k=3)
    calls = []
    for schedule in ("joint_rr", "joint_rrr"):
        model = build_model(data.X_train, data.y_train, cc, None)
        fr, _ = calibrate_free_radius(model, s_task=.5, snapshot_step=0)
        train_synchronized(model, data, cc, sync_cfg(schedule, "regression"), fr, near_scale=1., maintenance_epochs={2},
                           maintenance_hook=lambda ep, m, opt, ad, s=schedule: calls.append((s, ep)) or {})
    assert calls == [("joint_rrr", 2)]


@pytest.mark.parametrize("kw", [dict(joint_objective="mean"), dict(adapter_warmup_epochs=4),
                                dict(adapter_warmup_epochs=-1)])
def test_joint_config_validation(kw):
    with pytest.raises(ValueError):
        sync_cfg("joint_rr", **kw).validate("classification")


def test_warmup_is_joint_only():
    with pytest.raises(ValueError):
        sync_cfg("alternating_rr", adapter_warmup_epochs=1).validate("classification")


@pytest.mark.parametrize("mcb", [False, True])
def test_every_epoch_observer_never_changes_training(mcb):
    data = classification_data(40)
    cc = CoreConfig(task_type="classification", representation="mlp", mcb_enabled=mcb, normalize_embeddings=True,
                    epochs=40, patience=2, batch_size=8, top_k=4, seed=3)
    plain = build_model(data.X_train, data.y_train, cc, 2)
    observed = copy.deepcopy(plain)
    a = train_retrieval(plain, data.X_train, data.y_train, data.X_val, data.y_val, cc)
    seen = []
    b = train_retrieval(observed, data.X_train, data.y_train, data.X_val, data.y_val, cc,
                        checkpoint_hook=lambda ep, m, opt: seen.append(ep) or {"observed": ep},
                        checkpoint_epochs=EveryEpoch())
    assert a.epochs_run == b.epochs_run < cc.epochs  # early stopping still happens
    assert a.best_epoch == b.best_epoch
    assert seen == list(range(1, b.epochs_run + 1))
    assert all(torch.equal(v, observed.state_dict()[k]) for k, v in plain.state_dict().items())


def ablation_cfg():
    cfg = load_config(Path("configs/t1/ema_ablation_20261010.yaml"))
    cfg["core"].update(epochs=3, patience=3, batch_size=4, top_k=3, bias_init_k=3, mlp_dims=[4])
    cfg["reuse"].update(epochs=2, patience=2)
    cfg["sync"].update(epochs=4, patience=4)
    return cfg


def test_encoder_ablation_records_both_settings_per_arm(tmp_path):
    cfg = ablation_cfg()
    cfg["conditions"] = [{"schedule": "joint_rr"}, {"name": "joint_scratch", "schedule": "joint_rr", "core_pretrain": False,
                                                      "retrieval_lr_scale": 1.0}]
    runner = Runner(cfg, tmp_path)
    data = classification_data()
    runner.data_for = lambda ds, seed: data
    runner.run_encoder_ablation({"name": "tiny", "task_type": "classification"}, 30)
    rows = [json.loads(line) for line in next(tmp_path.rglob("runs__tiny.jsonl")).read_text().splitlines()]
    arms = [a["name"] for a in cfg["arms"]]
    labels = {r["condition"] for r in rows}
    for arm in arms:
        assert {f"{arm}__retrieval_only", f"{arm}__nominal_residual_scores_combined",
                f"{arm}__joint_rr", f"{arm}__joint_scratch"} <= labels
    for r in rows:
        assert r["arm"] in arms
        if r["condition"].endswith("retrieval_only"):
            assert {"test_accuracy_pre", "val_accuracy_pre", "core_mean_neighborhood_churn"} <= r.keys()
        else:
            assert {"test_accuracy_post", "val_accuracy_post"} <= r.keys()
    ema = [r for r in rows if r.get("mcb_enabled") and r["condition"].endswith("retrieval_only")]
    assert ema and all(r["online_ema_gap"] >= 0 for r in ema)
    raw = next(r for r in rows if r["arm"] == "raw" and r["condition"].endswith("retrieval_only"))
    assert raw["core_mean_representation_drift"] == 0.0
    scratch = [json.loads(p.read_text()) for p in tmp_path.rglob("run_manifest.json") if "joint_scratch" in str(p)]
    assert scratch and all(m["budgets"]["core_pretrain"] is False and m["budgets"]["core_epochs_run"] == 0 for m in scratch)


def test_sync_arms_keep_unarmed_labels(tmp_path):
    cfg = ablation_cfg()
    cfg.update(type="sync", conditions=[{"schedule": "joint_rr"}])
    cfg.pop("arms")
    runner = Runner(cfg, tmp_path)
    data = regression_data()
    runner.data_for = lambda ds, seed: data
    runner.run_sync({"name": "tiny", "task_type": "regression"}, 30)
    rows = [json.loads(line) for line in next(tmp_path.rglob("runs__tiny.jsonl")).read_text().splitlines()]
    assert [r["condition"] for r in rows] == ["joint_rr"]
    assert "arm" not in rows[0] and "val_rmse_post" in rows[0]
