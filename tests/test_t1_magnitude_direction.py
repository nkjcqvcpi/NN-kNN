"""Contracts of direction normalization, neighborhood evidence and learning signals."""

import copy

import pytest
import torch
from torch.nn import functional as F

from model.common.nominal import NominalSchema
from model.t1.magnitude_direction import (
    MagnitudeDirectionClassificationAdapter,
    MagnitudeDirectionNominalClassificationAdapter,
    MagnitudeDirectionRegressionAdapter,
    make_magnitude_direction_adapter,
    neighborhood_statistics,
    unit_direction,
)


def neighborhood():
    # Self is ineligible; nearest eligible case has zero activation (e.g. top-k
    # rejected it because its learned case bias is low).
    return {"weights": torch.tensor([[0., .25, .75, 0.]]),
            "distances": torch.tensor([[0., .7, .2, .1]]),
            "excluded": torch.tensor([[True, False, False, False]])}


def test_statistics_use_actual_distances_activation_weights_and_nonself_minimum():
    r = neighborhood()
    actual = neighborhood_statistics(r, torch.tensor([100., 2., 6., 9.]), "regression")
    torch.testing.assert_close(actual, torch.tensor([[.1, .325, .625, 3.]]))
    # Changing a self label cannot reveal the query target through these inputs.
    other = neighborhood_statistics(r, torch.tensor([-999., 2., 6., 9.]), "regression")
    torch.testing.assert_close(actual, other)


def test_classification_disagreement_is_gini_and_statistics_keep_retrieval_gradients():
    r = neighborhood()
    r["weights"].requires_grad_()
    r["distances"].requires_grad_()
    targets = F.one_hot(torch.tensor([0, 0, 1, 0]), 2).float()
    stats = neighborhood_statistics(r, targets, "classification")
    p0 = r["weights"] @ targets
    torch.testing.assert_close(stats[:, 3], 1 - p0.square().sum(1))
    stats.sum().backward()
    assert torch.isfinite(r["weights"].grad).all()
    assert torch.isfinite(r["distances"].grad).all()
    assert r["distances"].grad[0, 0] == 0  # excluded self
    assert r["distances"].grad[0, 3] == 1  # nearest eligible, even at zero weight


@pytest.mark.parametrize("bad", ["unnormalized", "negative", "nonfinite", "self_weight", "all_excluded"])
def test_invalid_retrieval_statistics_are_refused(bad):
    r = neighborhood()
    if bad == "unnormalized":
        r["weights"] *= 2
    elif bad == "negative":
        r["weights"][0, 1:3] = torch.tensor([-.2, 1.2])
    elif bad == "nonfinite":
        r["distances"][0, 2] = float("nan")
    elif bad == "self_weight":
        r["weights"][0] = torch.tensor([.25, 0., .75, 0.])
    else:
        r["excluded"].fill_(True)
    with pytest.raises(ValueError):
        neighborhood_statistics(r, torch.arange(4).float(), "regression")


def test_unit_direction_has_exact_scale_separation_and_explicit_zero_convention():
    raw = torch.tensor([[3., -4.], [0., 0.], [3e-30, -4e-30], [3e30, -4e30]])
    direction = unit_direction(raw)
    expected = torch.tensor([[.6, -.8], [1., 0.], [.6, -.8], [.6, -.8]])
    torch.testing.assert_close(direction, expected)
    torch.testing.assert_close(direction.norm(dim=1), torch.ones(4))
    torch.testing.assert_close(unit_direction(raw[:1] * 7), direction[:1])


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_model_scale_is_actual_correction_norm_and_state_roundtrips(task):
    torch.manual_seed(9)
    adapter = make_magnitude_direction_adapter(task, 3, num_classes=3 if task == "classification" else None,
                                             hidden_dims=(8, 4), initial_magnitude=.01)
    p0 = torch.full((2, 3), 1 / 3) if task == "classification" else torch.tensor([[.4], [.8]])
    dz, stats = torch.randn(2, 3), torch.tensor([[.1, .3, .4, .2], [.2, .5, .3, .1]])
    parts = adapter.forward_parts(dz, p0, stats)
    torch.testing.assert_close(parts.direction.norm(dim=1), torch.ones(2))
    torch.testing.assert_close(parts.magnitude, torch.full((2, 1), .01))
    torch.testing.assert_close(parts.correction.norm(dim=1, keepdim=True), parts.magnitude)
    torch.testing.assert_close(parts.final - p0, parts.correction)
    restored = copy.deepcopy(adapter)
    restored.load_state_dict(adapter.state_dict())
    torch.testing.assert_close(restored.forward_parts(dz, p0, stats).final, parts.final)
    # Raw-direction rescaling cannot change the final correction size.
    head = adapter.adapt_net_final if task == "regression" else adapter.net
    with torch.no_grad():
        head.direction_head.weight.mul_(9)
        head.direction_head.bias.mul_(9)
    enlarged = adapter.forward_parts(dz, p0, stats)
    torch.testing.assert_close(enlarged.final, parts.final)
    torch.testing.assert_close(enlarged.magnitude, parts.magnitude)


def test_scalar_auxiliary_loss_trains_direction_when_final_loss_cannot():
    adapter = MagnitudeDirectionRegressionAdapter(1, hidden_dims=(4, 3), initial_magnitude=.1)
    with torch.no_grad():
        adapter.adapt_net_final.direction_head.weight.zero_()
        adapter.adapt_net_final.direction_head.bias.fill_(.8)
    p0, target = torch.zeros(2, 1), torch.full((2, 1), -2.)
    parts = adapter.forward_parts(torch.zeros(2, 1), p0, torch.zeros(2, 4))
    final_loss = F.mse_loss(parts.final, target)
    direction_parameter = adapter.adapt_net_final.direction_head.bias
    scale_parameter = adapter.adapt_net_final.scale_head.bias
    direction_grad, scale_grad = torch.autograd.grad(final_loss, (direction_parameter, scale_parameter), retain_graph=True)
    torch.testing.assert_close(direction_grad, torch.zeros_like(direction_grad))
    assert scale_grad.abs().sum() > 0
    proposal_loss = F.mse_loss(parts.raw_proposal, target - p0)
    proposal_grad, = torch.autograd.grad(proposal_loss, (direction_parameter,))
    assert (proposal_grad > 0).all()  # gradient descent moves the wrong-positive proposal downward
    assert all(not p.requires_grad for p in adapter.adapt_net_pair.parameters())


def test_magnitude_head_uses_neighborhood_evidence_and_shared_adaptation_context():
    adapter = MagnitudeDirectionRegressionAdapter(1, hidden_dims=(3, 2))
    net = adapter.adapt_net_final
    with torch.no_grad():
        for layer in net.trunk:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(1)
                layer.bias.fill_(1)
        net.scale_head.weight.fill_(.1)
    dz, p0, stats = torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 4)
    base = adapter.forward_parts(dz, p0, stats)
    for col in range(4):
        evidence = stats.clone()
        evidence[0, col] = 1
        changed = adapter.forward_parts(dz, p0, evidence)
        assert changed.magnitude > base.magnitude
        torch.testing.assert_close(changed.raw_proposal, base.raw_proposal)
    assert adapter.forward_parts(dz + 1, p0, stats).magnitude > base.magnitude


def test_nominal_variant_preserves_stable_case_id_binding():
    schema = NominalSchema([{"name": "colour", "covered_by_representation": False}]).fit(
        {"colour": ["red", "blue"]})
    values = schema.encode({"colour": ["red", "blue"]})
    adapter = make_magnitude_direction_adapter("classification", 2, num_classes=2, hidden_dims=(4, 3),
        schema=schema, case_ids=torch.tensor([20, 10]), case_values=values)
    assert isinstance(adapter, MagnitudeDirectionNominalClassificationAdapter)
    torch.testing.assert_close(adapter.values_for_ids(torch.tensor([10, 20])), values.flip(0))
    parts = adapter.forward_parts(torch.zeros(1, 2), torch.full((1, 2), .5), torch.zeros(1, 4),
                                 torch.zeros(1, schema.dimension))
    assert parts.final.shape == (1, 2)
    with pytest.raises(ValueError, match="nominal"):
        adapter.forward_parts(torch.zeros(1, 2), torch.full((1, 2), .5), torch.zeros(1, 4))


def test_new_design_refuses_silent_legacy_calls_and_incompatible_output_modes():
    ad = MagnitudeDirectionClassificationAdapter(2, 2, hidden_dims=(4, 3))
    with pytest.raises(ValueError, match="neighborhood_statistics"):
        ad(torch.zeros(1, 2), torch.full((1, 2), .5))
    with pytest.raises(ValueError, match="logit_residual"):
        MagnitudeDirectionClassificationAdapter(2, 2, output_mode="logit_residual")
    with pytest.raises(ValueError, match="initial_magnitude"):
        MagnitudeDirectionRegressionAdapter(2, initial_magnitude=0)
