"""Tuning options of the 2026-10-13 pre-registration (docs/T1_PREREG_20261013.md).

Retrieval guard, error-based early stopping, score scale, adapter weight decay
and dropout, validation-fitted temperature, fixed-metric input locality, the
runner's ``splits.max_train`` fallback and CPU-only checkpoints. Every option
defaults to the earlier behavior; the first test pins that bit for bit.
"""
import copy
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import model.t1.reuse as reuse
import model.t1.sync as sync
import tools.t1_run as t1_run
from model.common.core import CoreConfig, build_model, train_retrieval
from model.nn_cdh import ClassificationNNCDHAdapter
from model.t1.adapted_retraining import continued_adapted_trial
from model.t1.calibration import calibrate_free_radius
from model.t1.data import T1Data
from model.t1.reuse import (ReuseConfig, _adapter_losses, adapter_optimizer, calibrate_temperature, calibration_logits,
                            evaluate_regression_reuse, evaluate_reuse, fit_temperature, separated,
                            train_classification_adapter, train_regression_adapter)
from model.t1.sync import SyncConfig, _forward_terms, train_synchronized
from tools.t1_run import Runner, input_locality, load_config, to_cpu


# --------------------------------------------------------------------------- toy scenarios


def toy_classification():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(60, 3, generator=g)
    y = (X[:, 0] > -.4).long() + (X[:, 0] + .3 * X[:, 1] > .4).long()
    return T1Data(name="toy", task_type="classification", X_train=X[:36], y_train=y[:36],
                  X_val=X[36:48], y_val=y[36:48], X_test=X[48:], y_test=y[48:], num_classes=3)


def toy_regression():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(48, 2, generator=g)
    y = X[:, 0] - .5 * X[:, 1] ** 2 + .1 * torch.randn(48, generator=g)
    return T1Data(name="toy", task_type="regression", X_train=X[:30], y_train=y[:30],
                  X_val=X[30:39], y_val=y[30:39], X_test=X[39:], y_test=y[39:])


def checksum(module):
    return [float(v.double().sum()) for v in module.state_dict().values() if v.is_floating_point()]


def sync_scenario(task, schedule, representation="raw", **kw):
    data = toy_classification() if task == "classification" else toy_regression()
    cc = CoreConfig(task_type=task, batch_size=8, top_k=4, bias_init_k=4, representation=representation, seed=3)
    model = build_model(data.X_train, data.y_train, cc, data.num_classes)
    fr, _ = calibrate_free_radius(model, s_task=.5, snapshot_step=0)
    mode = "aggregate_regression" if task == "regression" else kw.pop("output_mode", "nominal_residual_scores")
    cfg = SyncConfig(schedule, 1., .5, .1, 1., 1., .1, output_mode=mode, hidden_dims=(8, 4), seed=1,
                     **{"epochs": 6, "patience": 6, "block_epochs": 2, **kw})
    adapter, info = train_synchronized(model, data, cc, cfg, fr, near_scale=1.)
    h = info["history"]
    return {"val_L_post": [r["val_L_post"] for r in h], "val_L_pre": [r["val_L_pre"] for r in h],
            "L_post": [r["L_post"] for r in h], "L_pre": [r["L_pre"] for r in h],
            "best_epoch": info["best_epoch"], "best_val_L_post": info["best_val_L_post"],
            "adapter": checksum(adapter), "model": checksum(model)}


def trained_core(task):
    data = toy_classification() if task == "classification" else toy_regression()
    cc = CoreConfig(task_type=task, batch_size=8, top_k=4, bias_init_k=4, epochs=5, patience=5, seed=3)
    model = build_model(data.X_train, data.y_train, cc, data.num_classes)
    tr = train_retrieval(model, data.X_train, data.y_train, data.X_val, data.y_val, cc)
    return data, cc, model, tr


def reuse_cfg(output_mode="nominal_residual_scores", **kw):
    return ReuseConfig(output_mode, "combined", 1., 1., "softmax", hidden_dims=(8, 4), **{"epochs": 12, "patience": 3,
                       "batch_size": 8, "seed": 2, **kw})


def reuse_scenario(output_mode):
    data, cc, model, tr = trained_core("classification")
    rc = reuse_cfg(output_mode)
    adapter, info = train_classification_adapter(model, data, rc)
    ev = evaluate_reuse(model, adapter, data.X_test, data.y_test, rc)
    out = {"val_loss": [r["val_loss"] for r in info["history"]], "best_epoch": info["best_epoch"],
           **{k: ev[k] for k in ("accuracy_pre", "accuracy_post", "ece_pre", "ece_post", "correction_l1_mean")},
           "adapter": checksum(adapter)}
    cont = continued_adapted_trial(model, tr.optimizer, adapter, info["optimizer_state"], data, cc, rc,
                                   retrieval_epochs=1, adapter_epochs=1, lr_scale=1.)
    out["continuation_loss"] = [r["loss"] for r in cont[4]["history"]]
    out["continuation_adapter"] = checksum(cont[2])
    return out


def regression_reuse_scenario():
    data, cc, model, tr = trained_core("regression")
    rc = reuse_cfg()
    adapter, info = train_regression_adapter(model, data, rc)
    ev = evaluate_regression_reuse(model, adapter, data.X_test, data.y_test, success_tolerance=.5)
    cont = continued_adapted_trial(model, tr.optimizer, adapter, info["optimizer_state"], data, cc, rc,
                                   retrieval_epochs=1, adapter_epochs=1, lr_scale=1.)
    return {"val_loss": [r["val_loss"] for r in info["history"]], "best_epoch": info["best_epoch"],
            "rmse_post": ev["rmse_post"], "adapter": checksum(adapter),
            "continuation_loss": [r["loss"] for r in cont[4]["history"]]}


def accelerator():
    return "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else None


@pytest.fixture
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# Captured with the pre-change source (commit 6bc0428), torch 2.14.0 on CPU, one thread:
# SHA-256 of json.dumps(outputs, sort_keys=True) plus readable headline values.
PRE_CHANGE_DIGEST = "75eac8233eeb38c61ac7f18567e82db1742e9aaa434decc5075cffb881a15410"
PRE_CHANGE_HEADLINES = {
    "sync_cls_alternating": (4, 1.0877645015716553), "sync_cls_independent": (6, 1.0973247289657593),
    "sync_cls_joint_mlp_logit": (6, 0.38731637597084045), "sync_reg_joint": (6, 0.2165873944759369),
    "sync_reg_alternating": (8, 0.23687344789505005), "reuse_scores": (12, 0.75), "reuse_logit": (1, 0.8333333134651184),
    "reuse_regression": (12, 0.33790048132837025)}


def test_defaults_reproduce_pre_change_toy_outputs_exactly(one_thread):
    out = {
        "sync_cls_alternating": sync_scenario("classification", "alternating_rr"),
        "sync_cls_independent": sync_scenario("classification", "independent", patience=1),
        "sync_cls_joint_mlp_logit": sync_scenario("classification", "joint_rr", "mlp", output_mode="logit_residual",
                                                  joint_objective="summed", adapter_warmup_epochs=1),
        "sync_reg_joint": sync_scenario("regression", "joint_rr"),
        "sync_reg_alternating": sync_scenario("regression", "alternating_rr", patience=2, epochs=8),
        "reuse_scores": reuse_scenario("nominal_residual_scores"),
        "reuse_logit": reuse_scenario("logit_residual"),
        "reuse_regression": regression_reuse_scenario(),
    }
    headlines = {k: (v["best_epoch"], v.get("best_val_L_post", v.get("accuracy_post", v.get("rmse_post"))))
                 for k, v in out.items()}
    assert headlines == PRE_CHANGE_HEADLINES
    text = json.dumps(out, sort_keys=True)
    assert hashlib.sha256(text.encode()).hexdigest() == PRE_CHANGE_DIGEST, text


# --------------------------------------------------------------------------- synchronization


def regression_setup():
    X = torch.tensor([[0.], [.1], [.3], [.8], [.9], [1.2]])
    y = torch.tensor([0., .1, .2, .8, 1., 1.2])
    cc = CoreConfig(task_type="regression", batch_size=3, top_k=3)
    model = build_model(X, y, cc, None)
    # five validation rows: validation passes are the only 5-row forward passes
    data = SimpleNamespace(X_train=X, y_train=y, X_val=X[:5] + .01, y_val=y[:5], num_classes=None)
    fr, _ = calibrate_free_radius(model, s_task=.5, snapshot_step=0)
    return model, data, cc, fr


def scripted_validation(monkeypatch, values):
    """Replace validation terms: call 0 is the pre-loop epoch-0 pass, call e is epoch e.

    Returns the call counter; reset it to 0 before each run.
    """
    original, calls = sync._forward_terms, [0]

    def terms(m, ad, x, target, *args, **kwargs):
        out = original(m, ad, x, target, *args, **kwargs)
        if len(x) == 5:
            out.update({k: torch.tensor(float(v)) for k, v in values(calls[0]).items()})
            calls[0] += 1
        return out
    monkeypatch.setattr(sync, "_forward_terms", terms)
    return calls


def reg_cfg(**kw):
    return SyncConfig("alternating_rr", 1., .5, .1, 1., 1., .1, output_mode="aggregate_regression",
                      **{"epochs": 8, "patience": 8, "block_epochs": 2, "hidden_dims": (8, 4), **kw})


def test_retrieval_guard_rejects_checkpoints_whose_pre_adaptation_loss_worsens(monkeypatch):
    # L_post improves every epoch; L_pre doubles from epoch 5 on.
    calls = scripted_validation(monkeypatch, lambda ep: {"L_post": 10 - ep, "L_pre": 2. if ep >= 5 else 1.})
    model, data, cc, fr = regression_setup()
    _, plain = train_synchronized(copy.deepcopy(model), data, cc, reg_cfg(patience=1), fr, near_scale=1.)
    assert plain["best_epoch"] == 8 and len(plain["history"]) == 8 and plain["guard_rejected_epochs"] == 0
    calls[0] = 0
    _, guarded = train_synchronized(copy.deepcopy(model), data, cc, reg_cfg(patience=1, retrieval_guard=.1), fr, near_scale=1.)
    hist = guarded["history"]
    assert guarded["v_pre0"] == 1.0
    assert guarded["best_epoch"] == 4 and guarded["best_val_L_post"] == 6.0
    # rejected epochs 5 and 6 still count toward patience 1
    assert [h["guard_rejected"] for h in hist] == [False] * 4 + [True] * 2
    assert guarded["guard_rejected_epochs"] == 2
    assert not plain["no_eligible_checkpoint"] and not guarded["no_eligible_checkpoint"]


def every_epoch_worsens_retrieval(ep):
    """Scripted validation: L_pre doubles after epoch 0, so a guard rejects every epoch."""
    return {"L_post": 10 - ep, "L_pre": 1. if ep == 0 else 2., "err_post": 3. if ep == 0 else 1.}


def test_guard_without_an_eligible_checkpoint_returns_the_epoch0_state(monkeypatch):
    model, data, cc, fr = regression_setup()
    initial_adapter, observed = [], []
    real_optimizer = sync.adapter_optimizer

    def recording_optimizer(params, *args):  # called once, right after the adapter is built
        params = list(params)
        initial_adapter.extend(p.detach().clone() for p in params)
        return real_optimizer(params, *args)

    monkeypatch.setattr(sync, "adapter_optimizer", recording_optimizer)
    calls = scripted_validation(monkeypatch, every_epoch_worsens_retrieval)
    cfg = lambda **kw: SyncConfig("joint_rr", 1., .5, .1, 1., 1., .1, output_mode="aggregate_regression",  # noqa: E731
                                  epochs=6, patience=1, hidden_dims=(8, 4), **kw)
    _, plain = train_synchronized(copy.deepcopy(model), data, cc, cfg(), fr, near_scale=1.)
    assert plain["best_epoch"] == 6 and not plain["no_eligible_checkpoint"]
    calls[0], initial_adapter[:] = 0, []
    m = copy.deepcopy(model)
    start = copy.deepcopy(m.state_dict())
    adapter, info = train_synchronized(m, data, cc, cfg(retrieval_guard=.1), fr, near_scale=1.,
                                       epoch_observer=lambda ep, mdl, ad: observed.append(copy.deepcopy(mdl.state_dict())) or {})
    hist = info["history"]
    # epochs 1 and 2 are rejected and exhaust patience 1; none is eligible
    assert [h["guard_rejected"] for h in hist] == [True, True] and all(h["optimizer_updates"] > 0 for h in hist)
    assert info["no_eligible_checkpoint"] and info["best_epoch"] == 0 and info["optimizer_checkpoint_epoch"] == 0
    assert (info["best_val_L_post"], info["best_val_err"]) == (10., 3.)  # epoch-0 validation terms: finite
    assert any(not torch.equal(v, observed[-1][k]) for k, v in start.items())  # training did move retrieval
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in start.items())  # ... and the epoch-0 state is back
    assert all(torch.equal(a, b) for a, b in zip([p for p in adapter.parameters() if p.requires_grad], initial_adapter))
    assert info["adapter_optimizer_state"]["state"] == {} and info["retrieval_optimizer_state"]["state"] == {}
    assert not adapter.training


def test_guard_without_an_eligible_checkpoint_after_a_maintenance_edit_raises(monkeypatch):
    model, data, cc, fr = regression_setup()
    calls = scripted_validation(monkeypatch, every_epoch_worsens_retrieval)

    def hook(edit):
        def run(ep, mdl, opt, adapter):
            if edit:
                with torch.no_grad():
                    mdl.biases.add_(.01)
            return {}
        return run

    cfg = lambda **kw: SyncConfig("alternating_rrr", 1., .5, .1, 1., 1., .1, output_mode="aggregate_regression",  # noqa: E731
                                  epochs=6, patience=1, block_epochs=2, hidden_dims=(8, 4), **kw)
    kw = dict(near_scale=1., maintenance_epochs={3})
    with pytest.raises(RuntimeError, match="maintenance edit"):
        train_synchronized(copy.deepcopy(model), data, cc, cfg(retrieval_guard=.1), fr, maintenance_hook=hook(True), **kw)
    calls[0] = 0
    _, info = train_synchronized(copy.deepcopy(model), data, cc, cfg(retrieval_guard=.1), fr, maintenance_hook=hook(False), **kw)
    assert info["no_eligible_checkpoint"] and info["best_epoch"] == 0  # an audit-only maintenance keeps epoch 0
    assert info["history"][2]["maintenance_state_changed"] is False
    calls[0] = 0
    _, info = train_synchronized(copy.deepcopy(model), data, cc, cfg(), fr, maintenance_hook=hook(True), **kw)
    assert info["best_epoch"] > 0 and info["history"][2]["maintenance_state_changed"] is True


def test_val_error_selection_is_lexicographic(monkeypatch):
    err = {3: .5, 4: .3, 5: .4, 6: .3, 7: .3, 8: .6}
    post = {3: 5., 4: 4., 5: 1., 6: 3., 7: 3.5, 8: .5}
    script = lambda ep: {"err_post": err.get(ep, 1.), "L_post": post.get(ep, 9.)}  # noqa: E731
    model, data, cc, fr = regression_setup()
    calls = scripted_validation(monkeypatch, script)
    _, by_loss = train_synchronized(copy.deepcopy(model), data, cc, reg_cfg(), fr, near_scale=1.)
    calls[0] = 0
    _, by_err = train_synchronized(copy.deepcopy(model), data, cc, reg_cfg(early_stopping="val_error"), fr, near_scale=1.)
    assert by_loss["best_epoch"] == 8
    assert by_err["best_epoch"] == 6 and by_err["best_val_err"] == pytest.approx(.3) and by_err["best_val_L_post"] == 3.
    assert [h["val_err_post"] for h in by_err["history"]][2:] == pytest.approx([err[e] for e in range(3, 9)])


def test_forward_terms_report_no_grad_errors():
    data = toy_classification()
    cc = CoreConfig(task_type="classification", top_k=4, bias_init_k=4)
    model = build_model(data.X_train, data.y_train, cc, 3)
    adapter = ClassificationNNCDHAdapter(3, 3, 0, (8, 4), "nominal_residual_scores")
    T = _forward_terms(model, adapter, data.X_val, data.y_val, None, output_mode="nominal_residual_scores", near_scale=1.)
    r = model.retrieve(data.X_val, exclude_identical=False)
    w, labels = r["weights"], model.labels[r["case_indices"]].float()
    p0 = (w @ labels) / (w @ labels).sum(1, keepdim=True)
    _, s = adapter(r["query_features"] - w @ r["case_features"], p0)
    assert float(T["err_pre"]) == pytest.approx(float((p0.argmax(1) != data.y_val).float().mean()))
    assert float(T["err_post"]) == pytest.approx(float((s.argmax(1) != data.y_val).float().mean()))
    assert not T["err_pre"].requires_grad and not T["err_post"].requires_grad and T["L_post"].requires_grad
    reg = toy_regression()
    rm = build_model(reg.X_train, reg.y_train, CoreConfig(task_type="regression", top_k=4, bias_init_k=4), None)
    from model.nn_cdh import NNCDHAdapter
    R = _forward_terms(rm, NNCDHAdapter(2, 1, (8, 4)), reg.X_val, reg.y_val, None, output_mode="aggregate_regression", near_scale=1.)
    assert float(R["err_pre"]) == pytest.approx(math.sqrt(float(R["L_pre"].detach())))
    assert float(R["err_post"]) == pytest.approx(math.sqrt(float(R["L_post"].detach())))


def test_score_scale_multiplies_scores_inside_post_loss_only():
    data = toy_classification()
    model = build_model(data.X_train, data.y_train, CoreConfig(task_type="classification", top_k=4, bias_init_k=4), 3)
    adapter = ClassificationNNCDHAdapter(3, 3, 0, (8, 4), "nominal_residual_scores")
    kw = dict(output_mode="nominal_residual_scores", near_scale=1.)
    with torch.no_grad():
        T1 = _forward_terms(model, adapter, data.X_val, data.y_val, None, **kw)
        T3 = _forward_terms(model, adapter, data.X_val, data.y_val, None, score_scale=3., **kw)
        r = model.retrieve(data.X_val, exclude_identical=False)
        w, labels = r["weights"], model.labels[r["case_indices"]].float()
        p0 = (w @ labels) / (w @ labels).sum(1, keepdim=True)
        _, s = adapter(r["query_features"] - w @ r["case_features"], p0)
    assert float(T1["L_post"]) == pytest.approx(float(F.cross_entropy(s, data.y_val)))
    assert float(T3["L_post"]) == pytest.approx(float(F.cross_entropy(3. * s, data.y_val)))
    assert float(T3["L_post"]) != pytest.approx(float(T1["L_post"]))
    for key in ("L_pre", "L_near", "L_delta", "correction_norm", "err_post"):
        assert torch.equal(T1[key], T3[key])
    losses = _adapter_losses(adapter, s - p0, s, p0, data.y_val, reuse_cfg(score_scale=3.))
    assert float(losses["l_cls"]) == pytest.approx(float(F.cross_entropy(3. * s, data.y_val)))
    # a synchronized run selects on the scaled loss
    cc = CoreConfig(task_type="classification", batch_size=8, top_k=4, bias_init_k=4)
    m = build_model(data.X_train, data.y_train, cc, 3)
    ad, info = train_synchronized(m, data, cc, SyncConfig("joint_rr", 1., .5, .1, 1., 1., 0., epochs=3, patience=3,
                                  hidden_dims=(8, 4), score_scale=3.), None, near_scale=1.)
    m.eval()
    with torch.no_grad():
        V = _forward_terms(m, ad, data.X_val, data.y_val, None, score_scale=3., **kw)
    assert info["best_val_L_post"] == pytest.approx(float(V["L_post"]), abs=1e-6)


@pytest.mark.parametrize("kw,task", [
    (dict(score_scale=3., output_mode="logit_residual"), "classification"),
    (dict(score_scale=0.), "classification"), (dict(score_scale=float("nan")), "classification"),
    (dict(adapter_dropout=.1, output_mode="aggregate_regression"), "regression"),
    (dict(adapter_dropout=1.), "classification"), (dict(adapter_weight_decay=-1.), "classification"),
    (dict(early_stopping="val_accuracy"), "classification"), (dict(retrieval_guard=-.1), "classification")])
def test_sync_option_validation(kw, task):
    with pytest.raises(ValueError):
        SyncConfig("joint_rr", 1., .5, .1, 1., 1., .1, **kw).validate(task)


def test_reuse_option_validation_and_regression_rejects_dropout():
    for kw in (dict(score_scale=2., output_mode="logit_residual"), dict(score_scale=-1.), dict(dropout=1.),
               dict(weight_decay=-1.)):
        with pytest.raises(ValueError):
            reuse_cfg(**kw).validate()
    data, _, model, _ = trained_core("regression")
    for kw in (dict(dropout=.1), dict(score_scale=2.)):
        with pytest.raises(ValueError, match="classification"):
            train_regression_adapter(model, data, reuse_cfg(**kw))


def test_adamw_is_used_iff_weight_decay_is_positive():
    params = [torch.nn.Parameter(torch.ones(2))]
    assert type(adapter_optimizer(params, 1e-3, 0., "cpu")) is torch.optim.Adam
    opt = adapter_optimizer(params, 1e-3, .05, "cpu")
    assert type(opt) is torch.optim.AdamW and opt.param_groups[0]["weight_decay"] == .05
    assert adapter_optimizer(params, 1e-3, 0., "xpu").param_groups[0]["foreach"] is False
    data = toy_classification()
    cc = CoreConfig(task_type="classification", batch_size=8, top_k=4, bias_init_k=4)
    runs = {}
    for wd in (0., .05):
        model = build_model(data.X_train, data.y_train, cc, 3)
        adapter, info = train_synchronized(model, data, cc, SyncConfig("joint_rr", 1., .5, .1, 1., 1., 0., epochs=3, patience=3,
                                           hidden_dims=(8, 4), adapter_weight_decay=wd), None, near_scale=1.)
        group = info["adapter_optimizer_state"]["param_groups"][0]
        assert info["adapter_optimizer"] == ("AdamW" if wd else "Adam")
        assert group["weight_decay"] == wd and group["decoupled_weight_decay"] is bool(wd)
        runs[wd] = checksum(adapter)
    assert runs[0.] != runs[.05]
    data, cc, model, tr = trained_core("classification")
    for wd in (0., .05):
        rc = reuse_cfg(weight_decay=wd, epochs=2)
        adapter, info = train_classification_adapter(model, data, rc)
        assert info["optimizer_state"]["param_groups"][0]["decoupled_weight_decay"] is bool(wd)
        cont = continued_adapted_trial(model, tr.optimizer, adapter, info["optimizer_state"], data, cc, rc,
                                       retrieval_epochs=0, adapter_epochs=1, lr_scale=1.)
        assert type(cont[3]) is (torch.optim.AdamW if wd else torch.optim.Adam)


def test_classification_dropout_reaches_the_adapters():
    data = toy_classification()
    cc = CoreConfig(task_type="classification", batch_size=8, top_k=4, bias_init_k=4)
    model = build_model(data.X_train, data.y_train, cc, 3)
    adapter, _ = train_synchronized(model, data, cc, SyncConfig("alternating_rr", 1., .5, .1, 1., 1., 0., epochs=4,
                                    block_epochs=1, hidden_dims=(8, 4), adapter_dropout=.2), None, near_scale=1.)
    assert [m.p for m in adapter.modules() if isinstance(m, torch.nn.Dropout)] == [.2] and not adapter.training
    adapter, _ = train_classification_adapter(model, data, reuse_cfg(dropout=.3, epochs=2))
    assert [m.p for m in adapter.modules() if isinstance(m, torch.nn.Dropout)] == [.3]


# --------------------------------------------------------------------------- temperature


def test_temperature_lowers_validation_nll_and_never_changes_argmax():
    g = torch.Generator().manual_seed(4)
    y = torch.randint(0, 4, (200,), generator=g)
    clean = F.one_hot(y, 4).float() + .8 * torch.randn(200, 4, generator=g)
    for logits, direction in ((6. * clean, 1.), (.2 * clean, -1.)):  # overconfident, underconfident
        T, degenerate = calibrate_temperature(logits, y)
        assert T == fit_temperature(logits, y) and not degenerate and not separated(logits, y)
        nll = lambda t: float(F.cross_entropy(logits.double() / t, y))  # noqa: E731
        assert nll(T) < nll(1.) and math.copysign(1., T - 1.) == direction
        assert nll(T) <= min(nll(T * .9), nll(T * 1.1)) + 1e-9  # a minimum along T
        assert torch.equal((logits / T).softmax(1).argmax(1), logits.argmax(1))


def test_degenerate_temperature_fits_fall_back_to_one_and_are_flagged():
    g = torch.Generator().manual_seed(4)
    y = torch.randint(0, 4, (200,), generator=g)
    # Separated: no row is misranked, so the NLL keeps falling as T -> 0 and has no minimizer.
    separable = 3. * F.one_hot(y, 4).float() + .5 * torch.rand(200, 4, generator=g).clamp_max(.9)
    assert separated(separable, y) and calibrate_temperature(separable, y) == (1., True)
    assert fit_temperature(separable, y) == 1.
    nll = lambda z, yy, t: float(F.cross_entropy(z.double() / t, yy))  # noqa: E731
    assert nll(separable, y, .01) < nll(separable, y, .1) < nll(separable, y, 1.)  # what an unflagged fit would chase
    ties = torch.zeros(10, 3)  # constant NLL: no unique minimizer
    assert calibrate_temperature(ties, torch.zeros(10, dtype=torch.long)) == (1., True)
    # Not separated, but the minimizer (T ~ 0.004) lies below the lower bound 0.01.
    z = torch.tensor([[.05, 0.]] * 199 + [[1e-4, 0.]])
    t = torch.tensor([0] * 199 + [1])
    assert not separated(z, t) and nll(z, t, .004) < nll(z, t, .01)
    assert calibrate_temperature(z, t) == (1., True)
    # One clearly misranked row gives an interior minimizer: fitted, not flagged.
    z = torch.tensor([[2., 0.]] * 19 + [[1., 0.]])
    t = torch.tensor([0] * 19 + [1])
    T, degenerate = calibrate_temperature(z, t)
    assert not degenerate and .01 < T < 1 and nll(z, t, T) < nll(z, t, 1.)


def test_evaluate_reuse_temperature_metrics_leave_predictions_unchanged():
    data, _, model, _ = trained_core("classification")
    for mode in ("nominal_residual_scores", "logit_residual"):
        rc = reuse_cfg(mode, epochs=3)
        adapter, _ = train_classification_adapter(model, data, rc)
        z_val = reuse.reuse_logits(model, adapter, data.X_val, rc)
        T = fit_temperature(z_val, data.y_val)
        plain = evaluate_reuse(model, adapter, data.X_test, data.y_test, rc)
        scaled = evaluate_reuse(model, adapter, data.X_test, data.y_test, rc, temperature=T)
        assert "ece_post_T" not in plain and {k: v for k, v in scaled.items() if not k.endswith("_T")} == plain
        z = reuse.reuse_logits(model, adapter, data.X_test, rc)
        assert scaled["nll_post_T"] == pytest.approx(float(F.cross_entropy(z.double() / T, data.y_test)))
        assert float(F.cross_entropy(z_val.double() / T, data.y_val)) <= float(F.cross_entropy(z_val.double(), data.y_val))
        nb = reuse.neighborhood_inputs(model, data.X_test, exclude_identical=False)
        with torch.no_grad():
            _, s = adapter(nb["dz"], nb["p0"], nb["du"])
        assert torch.equal(z, calibration_logits(s, rc))
        expected = torch.log(s.clamp_min(1e-8)) if mode == "logit_residual" else s
        assert torch.equal(calibration_logits(s, rc), expected)


# --------------------------------------------------------------------------- locality


@pytest.mark.parametrize("representation", ["raw", "mlp"])
def test_input_locality_matches_a_manual_computation(representation):
    data = toy_classification()
    cc = CoreConfig(task_type="classification", top_k=4, bias_init_k=4, representation=representation)
    model = build_model(data.X_train, data.y_train, cc, 3)
    model.train()
    value = input_locality(model, data.X_test, batch_size=5)  # several batches
    assert model.training  # modes are restored
    model.eval()
    with torch.no_grad():
        r = model.retrieve(data.X_test, exclude_identical=False)
        cases = model.cases[r["case_indices"]]
        d = ((data.X_test[:, None, :] - cases[None]) ** 2).sum(-1).sqrt()
        manual = float((r["weights"] * d).sum(1).mean())
    assert value == pytest.approx(manual, rel=1e-6)
    assert (r["weights"] > 0).sum(1).max() <= 4  # top-k weights, as in prediction


# --------------------------------------------------------------------------- runner


def test_max_train_falls_back_to_the_splits_cap(monkeypatch):
    seen = []
    monkeypatch.setattr(t1_run, "make_splits", lambda *a, **kw: seen.append(kw["max_train"]))
    runner = Runner.__new__(Runner)
    runner.cfg = {"splits": {"test_frac": .2, "val_frac": .2, "max_train": 50}}
    runner.data_for({"name": "x", "task_type": "classification"}, 0)
    runner.data_for({"name": "x", "task_type": "classification", "max_train": 10}, 0)
    runner.cfg["splits"].pop("max_train")
    runner.data_for({"name": "x", "task_type": "classification"}, 0)
    assert seen == [50, 10, None]


def tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from tensors(v)


def test_to_cpu_is_identity_on_cpu_and_keeps_state_dict_metadata():
    state = build_model(torch.randn(6, 2), torch.tensor([0, 1, 0, 1, 0, 1]),
                        CoreConfig(task_type="classification", top_k=3), 2).state_dict()
    nested = {"model_state": state, "list": [torch.ones(2), (torch.zeros(1), 3)], "x": None}
    assert to_cpu(nested) is nested and to_cpu(state) is state
    device = accelerator()
    if device is None:
        pytest.skip("no accelerator for the device-to-CPU half")
    moved = to_cpu({"model_state": OrderedDict((k, v.to(device)) for k, v in state.items()),
                    "list": [torch.ones(2, device=device), (torch.zeros(1, device=device), 3)]})
    assert all(t.device.type == "cpu" for t in tensors(moved))
    assert isinstance(moved["list"][1], tuple) and moved["list"][1][1] == 3
    on_device = OrderedDict((k, v.to(device)) for k, v in state.items())
    on_device._metadata = state._metadata
    assert to_cpu(on_device)._metadata == state._metadata


def run_tiny_runner(tmp_path):
    """retrieval_only, two reuse and two sync conditions; returns (rows, manifests)."""
    cfg = load_config(Path("configs/t1/ema_ablation_20261010.yaml"))
    cfg["arms"] = [{"name": "raw", "representation": "raw", "mcb_enabled": False}]
    cfg["core"].update(epochs=3, patience=3, batch_size=4, top_k=3, bias_init_k=3)
    cfg["reuse"].update(epochs=2, patience=2)
    cfg["sync"].update(epochs=4, patience=4)
    cfg["reuse_conditions"] = [{"output_mode": "nominal_residual_scores", "loss": "combined"},
                               {"name": "scaled", "output_mode": "nominal_residual_scores", "loss": "combined",
                                "score_scale": 3.0, "weight_decay": 0.01, "dropout": 0.1}]
    cfg["conditions"] = [{"schedule": "joint_rr"},
                         {"name": "guarded", "schedule": "joint_rr", "retrieval_guard": 0.02, "early_stopping": "val_error",
                          "score_scale": 3.0, "adapter_weight_decay": 0.01, "adapter_dropout": 0.1, "hidden_dims": [8, 4]}]
    runner = Runner(cfg, tmp_path)
    data = toy_classification()
    runner.data_for = lambda ds, seed: data
    runner.run_encoder_ablation({"name": "toy", "task_type": "classification"}, 20)
    rows = {r["condition"]: r for r in map(json.loads, next(tmp_path.rglob("runs__toy.jsonl")).read_text().splitlines())}
    assert set(rows) == {"raw__retrieval_only", "raw__nominal_residual_scores_combined", "raw__scaled", "raw__joint_rr", "raw__guarded"}
    manifests = {p.parent.parent.name: json.loads(p.read_text()) for p in tmp_path.rglob("run_manifest.json")}
    assert set(manifests) == set(rows)
    return rows, manifests


def test_runner_records_new_metrics_flows_yaml_options_and_saves_cpu_checkpoints(tmp_path, monkeypatch):
    # Every torch.save must receive an object returned by to_cpu (a no-op on CPU,
    # so the CPU-tensor check below cannot see the wiring by itself).
    converted, saved = [], []
    real_to_cpu, real_save = t1_run.to_cpu, torch.save

    def spy_to_cpu(obj):
        converted.append(real_to_cpu(obj))
        return converted[-1]

    def spy_save(obj, f, *args, **kwargs):
        saved.append((obj, Path(f).name))
        return real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(t1_run, "to_cpu", spy_to_cpu)
    monkeypatch.setattr(torch, "save", spy_save)
    rows, manifests = run_tiny_runner(tmp_path)
    monkeypatch.undo()
    assert {name for _, name in saved} >= {"checkpoint.pt", "calibration_snapshots.pt", "data_snapshot_s20.pt"}
    assert all(any(obj is c for c in converted) for obj, _ in saved), [name for obj, name in saved
                                                                      if not any(obj is c for c in converted)]
    base = rows["raw__retrieval_only"]["test_locality_input"]
    assert base > 0 and rows["raw__nominal_residual_scores_combined"]["test_locality_input"] == base
    for label, row in rows.items():
        if label != "raw__retrieval_only":
            assert {"test_locality_input", "val_temperature", "val_temperature_degenerate", "test_ece_post_T",
                    "test_nll_post_T"} <= row.keys()
            assert row["val_temperature"] > 0 and "test_temperature" not in row
            assert isinstance(row["val_temperature_degenerate"], bool)
            assert row["val_temperature_degenerate"] is False or row["val_temperature"] == 1.
    assert {"best_val_err", "sync_val_L_pre_epoch0", "sync_guard_rejected_epochs"} <= rows["raw__guarded"].keys()
    for label in ("raw__joint_rr", "raw__guarded"):
        assert isinstance(rows[label]["sync_no_eligible_checkpoint"], bool)
        assert math.isfinite(rows[label]["best_val_L_post"]) and math.isfinite(rows[label]["best_val_err"])
    sync_desc, reuse_desc = manifests["raw__guarded"]["model"]["sync"], manifests["raw__scaled"]["model"]["adapter"]
    assert {k: sync_desc[k] for k in ("retrieval_guard", "early_stopping", "score_scale", "adapter_weight_decay",
                                      "adapter_dropout", "hidden_dims")} == dict(
        retrieval_guard=.02, early_stopping="val_error", score_scale=3., adapter_weight_decay=.01, adapter_dropout=.1,
        hidden_dims=[8, 4])
    assert {k: reuse_desc[k] for k in ("score_scale", "weight_decay", "dropout")} == dict(score_scale=3., weight_decay=.01, dropout=.1)
    assert manifests["raw__joint_rr"]["model"]["sync"]["retrieval_guard"] is None
    assert all(m["run"]["deterministic"] is False for m in manifests.values())
    files = list(tmp_path.rglob("*.pt"))
    assert any(p.name == "checkpoint.pt" for p in files) and any(p.name == "calibration_snapshots.pt" for p in files)
    for path in files:
        assert all(t.device.type == "cpu" for t in tensors(torch.load(path, weights_only=False)))


@pytest.fixture
def on_accelerator(monkeypatch):
    """Build NN-kNN models on the accelerator, as NNKNN_DEVICE=<device> does (skips without one)."""
    import model.nnknn_model as nnknn_model
    device = accelerator()
    if device is None:
        pytest.skip("no XPU or CUDA device")
    monkeypatch.setattr(nnknn_model, "device", torch.device(device))
    return device


def test_locality_and_temperature_metrics_run_on_an_accelerator(on_accelerator):
    # Arc GPUs lack float64: these metrics must not do float64 arithmetic on the device.
    data, cc, model, _ = trained_core("classification")
    assert model.cases.device.type == on_accelerator
    value = input_locality(model, data.X_test, batch_size=5)
    with torch.no_grad():
        model.eval()
        r = model.retrieve(data.X_test.to(model.cases.device), exclude_identical=False)
        d = torch.cdist(data.X_test.to(model.cases.device), model.cases[r["case_indices"]],
                        compute_mode="donot_use_mm_for_euclid_dist")
        manual = float((r["weights"] * d).sum(1).mean().cpu())
    assert math.isfinite(value) and value == pytest.approx(manual, rel=1e-5)
    for mode in ("nominal_residual_scores", "logit_residual"):
        rc = reuse_cfg(mode, epochs=2)
        adapter, _ = train_classification_adapter(model, data, rc)
        z_val = reuse.reuse_logits(model, adapter, data.X_val, rc)
        T, degenerate = calibrate_temperature(z_val, data.y_val)
        scaled = evaluate_reuse(model, adapter, data.X_test, data.y_test, rc, temperature=.5)
        z = reuse.reuse_logits(model, adapter, data.X_test, rc).cpu().double()
        assert scaled["nll_post_T"] == pytest.approx(float(F.cross_entropy(z / .5, data.y_test)))
        assert T > 0 and isinstance(degenerate, bool) and math.isfinite(scaled["ece_post_T"])


def test_runner_on_an_accelerator_saves_checkpoints_that_load_on_the_cpu(tmp_path, on_accelerator):
    rows, manifests = run_tiny_runner(tmp_path)
    assert all(m["run"]["device"].startswith(on_accelerator) for m in manifests.values())
    assert all(math.isfinite(row["test_locality_input"]) for row in rows.values())
    assert all(math.isfinite(row["test_nll_post_T"]) for label, row in rows.items() if label != "raw__retrieval_only")
    files = list(tmp_path.rglob("*.pt"))
    assert {"checkpoint.pt", "calibration_snapshots.pt", "data_snapshot_s20.pt"} <= {p.name for p in files}
    for path in files:  # no map_location: tensors come back on the device they were saved from
        assert all(t.device.type == "cpu" for t in tensors(torch.load(path, weights_only=False)))


def test_deterministic_environment_switch(monkeypatch, tmp_path):
    seen = []

    class Stub:
        def __init__(self, cfg, out):
            seen.append(torch.are_deterministic_algorithms_enabled())

        def __getattr__(self, name):
            return lambda ds, seed: None

    monkeypatch.setattr(t1_run, "Runner", Stub)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # restored after the test
    monkeypatch.setattr("sys.argv", ["t1_run.py", "configs/t1/smoke_sync.yaml", "--out", str(tmp_path), "--seeds", "0",
                                     "--only-dataset", "none"])
    try:
        monkeypatch.setenv("NNKNN_DETERMINISTIC", "1")
        t1_run.main()
        monkeypatch.delenv("NNKNN_DETERMINISTIC")
        torch.use_deterministic_algorithms(False)
        t1_run.main()
    finally:
        torch.use_deterministic_algorithms(False)
    assert seen == [True, False]


def test_qualified_dataset_entry_gets_a_role_qualified_key(tmp_path):
    from tools.t1_run import Runner, load_config
    cfg = load_config(Path("configs/t1/ema_ablation_20261010.yaml"))
    cfg["datasets"] = [{"name": "wine", "task_type": "regression", "qualified": True}]
    runner = Runner(cfg, tmp_path)
    assert runner.dataset_key(cfg["datasets"][0]) == "regression__wine"
    cfg["datasets"] = [{"name": "wine", "task_type": "classification"}]
    assert Runner(cfg, tmp_path).dataset_key(cfg["datasets"][0]) == "wine"
