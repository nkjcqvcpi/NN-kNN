import copy
import pytest
import torch
from torch import nn

from model.common import tracking
from model.t1 import records
from model.t1.disagreement_threshold import (
    DisagreementThresholdAdapter, adapter_loss, routing_costs, routing_loss, train_threshold,
)


def fixture_data(kind="regression"):
    d = torch.linspace(0, 1, 101)
    target = (d > .4).long()
    p0 = torch.tensor([[.6, .4]]).repeat(len(d), 1) if kind=="classification" else torch.zeros(len(d), 1)
    net = nn.Sequential(nn.Linear(1, p0.shape[1]))
    with torch.no_grad():
        net[0].weight.zero_()
        net[0].bias.copy_(torch.tensor([-1., 1.]) if kind=="classification" else torch.tensor([1.]))
    values = dict(x=d[:, None], p0=p0, statistics=torch.stack((d, d, d, d), 1),
                  target=target if kind=="classification" else target.float())
    return net, values


def settings():
    return dict(score_scale=1., v_post=1., v_delta=1., v_small=0., free_radius=None,
                lr=.001, weight_decay=0., epochs=5, patience=2, batch_size=64,
                threshold_lr=.05, threshold_steps=5)


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_one_parameter_learns_useful_cutoff_and_outputs_exact_branches(kind):
    network, values = fixture_data(kind)
    model = DisagreementThresholdAdapter(network, kind, "threshold_frozen")
    assert [name for name, p in model.named_parameters() if p.requires_grad] == ["threshold"]
    model.fit_normalizer(values)
    p = model(values["x"], values["p0"], values["statistics"])
    before, full = routing_costs(values["p0"], p.full, values["target"], kind)
    initial = float(torch.where(p.apply, full, before).mean())
    optimizer = torch.optim.Adam([model.threshold], lr=.05)
    for _ in range(100):
        optimizer.zero_grad()
        routing_loss(model.logits(values["statistics"]), before, full).backward()
        optimizer.step()
    p = model(values["x"], values["p0"], values["statistics"])
    assert .38 < float(model.raw_threshold.detach()) < .43
    assert float(torch.where(p.apply, full, before).mean()) < initial
    assert torch.equal(p.final, torch.where(p.apply[:, None], p.full, values["p0"]))
    assert all(torch.equal(v, model.network.state_dict()[k]) for k, v in network.state_dict().items())


def test_cost_gradient_prefers_large_gain_over_more_small_harms_and_detaches_costs():
    # A single location: the large gain outweighs two small harms.
    threshold = torch.tensor(0., requires_grad=True)
    before = torch.tensor([2., 0., 0.], requires_grad=True)
    full = torch.tensor([0., .1, .1], requires_grad=True)
    routing_loss(-threshold.expand(3), before, full).backward()
    assert threshold.grad > 0  # Descent lowers cutoff, applying more.
    assert before.grad is None and full.grad is None


def test_joint_adapter_learns_even_when_everything_is_skipped():
    net, values = fixture_data()
    model = DisagreementThresholdAdapter(net, "regression", "threshold_joint")
    model.fit_normalizer(values)
    with torch.no_grad():
        model.threshold.fill_(100)
    assert not model(values["x"], values["p0"], values["statistics"]).apply.any()
    adapter_loss(model, values, settings()).backward()
    assert model.network[0].bias.grad.abs().sum() > 0
    assert model.threshold.grad is None


def test_constant_disagreement_and_targets_do_not_change_normalizer():
    net, values = fixture_data()
    values["statistics"].fill_(0.)
    model = DisagreementThresholdAdapter(net, "regression", "threshold_frozen")
    model.fit_normalizer(values)
    state = copy.deepcopy(model.state_dict())
    values["target"].fill_(100.)
    output = model(values["x"], values["p0"], values["statistics"])
    assert torch.isfinite(output.logit).all()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


def test_training_requires_recorded_run():
    net, values = fixture_data()
    with pytest.raises(TypeError, match="recorded run"):
        train_threshold(DisagreementThresholdAdapter(net, "regression", "threshold_frozen"),
                        values, values, settings(), 40, run=None)


def test_frozen_training_ignores_validation_targets_and_keeps_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path))
    net, values = fixture_data()
    results = []
    for name, val_target in (("a", values["target"]), ("b", 1-values["target"])):
        model = DisagreementThresholdAdapter(net, "regression", "threshold_frozen")
        key = records.key("tt", name)
        with tracking.open_run(key, config={"name":name}) as run:
            info = train_threshold(model, values, {**values, "target":val_target}, settings(), 40, run=run)
        saved = tracking.load_unit(key, config={"name":name}, arrays=False)
        checkpoint = saved.load_state("checkpoint")
        assert checkpoint["epoch"] == settings()["epochs"] == info["selected_epoch"]
        assert (saved.info.path/"weights"/"resume"/"last.pt").exists()
        results.append(copy.deepcopy(model.state_dict()))
    assert all(torch.equal(v, results[1][k]) for k, v in results[0].items())


@pytest.mark.parametrize("mode", ["threshold_frozen", "threshold_joint"])
def test_interrupted_training_resumes_without_replaying_epochs(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path))
    net, values = fixture_data()
    expected = DisagreementThresholdAdapter(net, "regression", mode)
    with tracking.open_run(records.key("tt", "complete"), config={"mode":mode}) as run:
        info = train_threshold(expected, values, values, settings(), 40, run=run)
    actual = DisagreementThresholdAdapter(net, "regression", mode)
    key = records.key("tt", "resume")
    with pytest.raises(RuntimeError, match="test interruption"):
        with tracking.open_run(key, config={"mode":mode}) as run:
            original = run.save_resume
            def interrupt(*args, **kwargs):
                original(*args, **kwargs)
                if kwargs["step"] == 2:
                    raise RuntimeError("test interruption")
            monkeypatch.setattr(run, "save_resume", interrupt)
            train_threshold(actual, values, values, settings(), 40, run=run)
    with tracking.open_run(key, config={"mode":mode}) as run:
        resumed = train_threshold(actual, values, values, settings(), 40, run=run)
    assert info == resumed
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in expected.state_dict().items())


def test_summary_plot_accepts_a_single_task_family(tmp_path):
    from tools.t1_adaptation_threshold import METHODS, plot
    report = dict(families=[dict(task_type="classification", method=m, mean_error=.1, n_tasks=1)
                            for m in METHODS],
                  tasks=[dict(task_type="classification", n_seeds=3)])
    plot(report, tmp_path)
    assert (tmp_path/"threshold_comparison.png").stat().st_size > 0
