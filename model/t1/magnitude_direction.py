"""T1 adaptation with an explicit correction magnitude and unit direction.

The raw proposal has the historical NN-CDH inputs and a residual-fitting loss.
Only its unit direction enters the final prediction. A second head receives
the shared hidden representation and four retrieval statistics and predicts a
nonnegative correction magnitude. It is not a probability or confidence score.

Scalar normalization is a hard sign and blocks the final-loss gradient into
the direction head. The auxiliary loss MUST supervise ``raw_proposal`` (not
``correction``), and final prediction loss MUST train the magnitude head.
Classification operates in nominal class-score space, not logit-residual mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model.common.nominal import NominalClassificationAdapter
from model.nn_cdh import ClassificationNNCDHAdapter, NNCDHAdapter


STATISTIC_NAMES = ("nearest_distance", "weighted_mean_distance", "activation_concentration", "target_disagreement")


@dataclass(frozen=True)
class AdaptationParts:
    raw_proposal: torch.Tensor
    correction: torch.Tensor
    magnitude: torch.Tensor
    direction: torch.Tensor
    final: torch.Tensor

    @property
    def scale(self) -> torch.Tensor:
        """Alias for the actual L2 magnitude of the applied correction."""
        return self.magnitude


def neighborhood_statistics(retrieval: dict, labels: torch.Tensor, task_type: str) -> torch.Tensor:
    """Return [nearest distance, weighted distance, sum(w**2), target variance].

    Distances are the actual learned retrieval metric, and weights are the
    normalized query-specific case activations. The nearest case is chosen
    over ALL eligible cases in this retrieval event, including zero-activation
    cases, excluding self/quarantined cases via ``retrieval['excluded']``.
    It is not necessarily the nearest retained top-k case.

    Labels must be aligned with the selected case columns (not the whole
    original memory). Regression labels are in training-standardized units.
    For one-hot classification targets the variance equals 1 - sum(p0**2).
    The general variance also works for valid soft class-mass labels.
    No query target, case bias, or detached proxy distance is used.
    """
    if task_type not in {"classification", "regression"}:
        raise ValueError("statistics require classification or regression task_type")
    w, d = retrieval["weights"], retrieval["distances"]
    if w.ndim != 2 or w.shape != d.shape or w.shape[1] < 1:
        raise ValueError("retrieval distances and weights must be aligned [queries, cases]")
    if not torch.isfinite(w).all() or not torch.isfinite(d).all() or (w < 0).any() or (d < 0).any():
        raise ValueError("retrieval distances and activations must be finite and nonnegative")
    if not torch.allclose(w.sum(1), torch.ones(w.shape[0], device=w.device, dtype=w.dtype), atol=1e-5):
        raise ValueError("retrieval case activations must already be normalized")
    excluded = retrieval.get("excluded")
    if excluded is None:
        eligible = torch.ones_like(w, dtype=torch.bool)
    else:
        if excluded.shape != w.shape or excluded.dtype != torch.bool:
            raise ValueError("retrieval exclusion mask must be Boolean and align with activations")
        eligible = ~excluded.to(device=w.device)
        if w.masked_select(~eligible).ne(0).any():
            raise ValueError("excluded cases must have zero retrieval activation")
    if not eligible.any(1).all():
        raise ValueError("each query needs an eligible nonself retrieval case")
    t = labels.to(device=w.device, dtype=w.dtype)
    if t.ndim == 1 and task_type == "regression":
        t = t[:, None]
    if t.ndim != 2 or t.shape[0] != w.shape[1] or t.shape[1] < 1 or not torch.isfinite(t).all():
        raise ValueError("case targets must be finite [cases, target dimensions] aligned with retrieval")
    if task_type == "classification":
        if (t < 0).any() or not torch.allclose(t.sum(1), torch.ones_like(t[:, 0]), atol=1e-5):
            raise ValueError("classification case targets must be one-hot or normalized class-mass vectors")
    mean_target = w @ t
    variance = (w @ t.square().sum(1) - mean_target.square().sum(1)).clamp_min(0)
    values = torch.stack((d.masked_fill(~eligible, float("inf")).amin(1),
                          (w * d).sum(1), w.square().sum(1), variance), dim=1)
    if not torch.isfinite(values).all():
        raise ValueError("neighborhood statistics overflowed or are nonfinite")
    return values


def unit_direction(proposal: torch.Tensor) -> torch.Tensor:
    """Exact unit direction for nonzero proposals; deterministic +e1 at zero.

    A zero vector has no direction. The +e1 convention preserves a unit output
    and hence correction norm == magnitude. It is an explicit tie convention,
    not evidence supporting a positive regression correction or class zero.
    Auxiliary residual supervision can move a zero proposal off this tie.
    Unlike epsilon normalization, tiny nonzero proposals retain unit norm.
    """
    # Scale first so very small/large finite proposals do not under/overflow
    # while squaring their components in an L2 norm.
    largest = proposal.abs().amax(dim=-1, keepdim=True)
    nonzero = largest > 0
    scaled = proposal / torch.where(nonzero, largest, torch.ones_like(largest))
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    denominator = torch.where(nonzero, norm, torch.ones_like(norm))
    fallback = torch.zeros_like(proposal)
    fallback[..., 0] = 1
    return torch.where(nonzero, scaled / denominator, fallback)


class _MagnitudeDirectionHead(nn.Module):
    def __init__(self, original_network: nn.Sequential, initial_magnitude: float):
        super().__init__()
        if not math.isfinite(initial_magnitude) or initial_magnitude <= 0:
            raise ValueError("initial_magnitude must be finite and strictly positive")
        children = list(original_network.children())
        if not children or not isinstance(children[-1], nn.Linear):
            raise ValueError("the shared NN-CDH trunk must end in a linear residual head")
        self.trunk = nn.Sequential(*children[:-1])
        self.direction_head = children[-1]
        self.scale_head = nn.Linear(self.direction_head.in_features + len(STATISTIC_NAMES), 1)
        # Stable inverse softplus; initial predictions apply a small correction.
        initial_bias = initial_magnitude + math.log(-math.expm1(-initial_magnitude))
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, initial_bias)

    def forward(self, inputs: torch.Tensor, statistics: torch.Tensor, p0: torch.Tensor) -> AdaptationParts:
        if statistics is None:
            raise ValueError("magnitude-direction adaptation requires actual neighborhood_statistics")
        if statistics.shape != (inputs.shape[0], len(STATISTIC_NAMES)) or not torch.isfinite(statistics).all():
            raise ValueError("neighborhood_statistics must be finite [queries, 4]")
        hidden = self.trunk(inputs)
        raw = self.direction_head(hidden)
        direction = unit_direction(raw)
        magnitude = F.softplus(self.scale_head(torch.cat((hidden, statistics.to(hidden)), dim=1)))
        correction = magnitude * direction
        return AdaptationParts(raw, correction, magnitude, direction, p0 + correction)


class MagnitudeDirectionRegressionAdapter(NNCDHAdapter):
    """Historical regression trunk; raw residual head plus magnitude head."""

    adapter_design = "magnitude_direction"

    def __init__(self, feature_dim: int, label_dim: int = 1, hidden_dims: tuple[int, int] = (16, 8),
                 *, initial_magnitude: float = .01):
        super().__init__(feature_dim, label_dim, hidden_dims)
        self.adapt_net_pair.requires_grad_(False)
        self.adapt_net_final = _MagnitudeDirectionHead(self.adapt_net_final.model, initial_magnitude)

    def forward_parts(self, delta_z, p0, neighborhood_statistics, delta_u=None) -> AdaptationParts:
        if delta_u is not None and delta_u.shape[-1] != 0:
            raise ValueError("regression adaptation has no nominal-difference input")
        return self.adapt_net_final(torch.cat((delta_z, p0), dim=1), neighborhood_statistics, p0)

    def forward_aggregate(self, query_features, case_features, case_labels, case_weights,
                          *, neighborhood_statistics=None):
        labels = case_labels.reshape(case_weights.shape[1], self.label_dim)
        p0 = case_weights @ labels
        return self.forward_parts(query_features - case_weights @ case_features,
                                  p0, neighborhood_statistics).final

    def forward(self, *args, **kwargs):
        raise ValueError("magnitude-direction adaptation uses the aggregate neighborhood, not per-case adaptation")

    def fit_pairs(self, *args, **kwargs):
        raise ValueError("magnitude-direction adaptation requires recorded aggregate T1 training")

    def fit_aggregate(self, *args, **kwargs):
        raise ValueError("magnitude-direction adaptation requires neighborhood statistics and recorded T1 training")


class MagnitudeDirectionClassificationAdapter(ClassificationNNCDHAdapter):
    """Shared classification trunk with a unit score direction and magnitude.

    The raw proposal is unconstrained and supervised toward one_hot(y) - p0.
    Its normalization is applied before scaling; no tanh magnitude remains in
    the final direction. A common class-score offset is not removed implicitly.
    """

    adapter_design = "magnitude_direction"

    def __init__(self, feature_dim: int, num_classes: int, nominal_dim: int = 0,
                 hidden_dims: tuple[int, int] = (64, 32), output_mode: str = "nominal_residual_scores",
                 dropout: float = 0., *, initial_magnitude: float = .01):
        _validate_classification_mode(output_mode)
        # Explicit base call also keeps the nominal multiple-inheritance path clear.
        ClassificationNNCDHAdapter.__init__(self, feature_dim, num_classes, nominal_dim, hidden_dims,
                                            output_mode, dropout)
        self.net = _MagnitudeDirectionHead(self.net, initial_magnitude)

    def forward_parts(self, delta_z, p0, neighborhood_statistics, delta_u=None) -> AdaptationParts:
        parts = [delta_z]
        if self.nominal_dim:
            if delta_u is None or delta_u.shape != (delta_z.shape[0], self.nominal_dim) or not torch.isfinite(delta_u).all():
                raise ValueError("uncovered nominal fields require actual finite query-minus-neighborhood differences")
            parts.append(delta_u)
        elif delta_u is not None and delta_u.shape[-1] != 0:
            raise ValueError("nominal fields covered by representation must not be duplicated")
        parts.append(p0)
        return self.net(torch.cat(parts, dim=1), neighborhood_statistics, p0)

    def forward(self, delta_z, p0, delta_u=None, *, neighborhood_statistics=None):
        parts = self.forward_parts(delta_z, p0, neighborhood_statistics, delta_u)
        return parts.correction, parts.final

    def compute_loss(self, *args, **kwargs):
        raise ValueError("use T1 losses with parts.raw_proposal for residual supervision and parts.final for task loss")


class MagnitudeDirectionNominalClassificationAdapter(MagnitudeDirectionClassificationAdapter,
                                                     NominalClassificationAdapter):
    """Magnitude/direction heads with the existing stable-ID nominal binding."""

    def __init__(self, *, schema, case_ids, case_values, initial_magnitude: float = .01, **kwargs):
        _validate_classification_mode(kwargs.get("output_mode", "nominal_residual_scores"))
        NominalClassificationAdapter.__init__(self, schema=schema, case_ids=case_ids,
                                              case_values=case_values, **kwargs)
        self.net = _MagnitudeDirectionHead(self.net, initial_magnitude)


def _validate_classification_mode(output_mode):
    if output_mode != "nominal_residual_scores":
        raise ValueError("magnitude-direction classification uses nominal_residual_scores, not logit_residual")


def make_magnitude_direction_adapter(task_type, feature_dim, *, num_classes=None,
                                     hidden_dims=(64, 32), output_mode="nominal_residual_scores", dropout=0.,
                                     initial_magnitude=.01, schema=None, case_ids=None, case_values=None):
    """Build the same design for both tasks, retaining explicit nominal binding."""
    if task_type == "regression":
        if schema is not None or dropout != 0:
            raise ValueError("regression magnitude-direction adaptation has no nominal fields or dropout")
        return MagnitudeDirectionRegressionAdapter(feature_dim, 1, hidden_dims,
                                                   initial_magnitude=initial_magnitude)
    if task_type != "classification" or num_classes is None:
        raise ValueError("classification requires num_classes; task_type must be classification or regression")
    kwargs = dict(feature_dim=feature_dim, num_classes=num_classes, hidden_dims=hidden_dims,
                  output_mode=output_mode, dropout=dropout, initial_magnitude=initial_magnitude)
    if schema is None:
        return MagnitudeDirectionClassificationAdapter(nominal_dim=0, **kwargs)
    return MagnitudeDirectionNominalClassificationAdapter(schema=schema, case_ids=case_ids,
                                                          case_values=case_values, **kwargs)
