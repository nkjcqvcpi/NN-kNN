"""Original NN-CDH residual adapters with the query representation appended.

With the prediction input retained, zero query connections preserve the initial
function and RNG state. A residual output does not prevent cancellation of
retrieval; query access is an explicit, opt-in relaxation of the T1 input rule.
The no-prediction variant removes the retrieval input columns from the residual
network, while retaining retrieval in the final additive skip connection.
"""
import copy
import torch
from torch import nn
from model.nn_cdh import ClassificationNNCDHAdapter, NNCDHAdapter

QUERY_AWARE_DESIGNS = frozenset({"query_aware", "query_aware_no_prediction"})


def is_query_aware(adapter):
    return getattr(adapter, "adapter_design", "residual") in QUERY_AWARE_DESIGNS


def append_zero_inputs(network, count):
    first = network[0]
    if not isinstance(first, nn.Linear) or count < 1:
        raise ValueError("a linear first layer and positive query dimension are required")
    expanded = copy.deepcopy(first)
    expanded.in_features += count
    expanded.weight = nn.Parameter(torch.cat((first.weight.detach().clone(),
        first.weight.new_zeros((first.out_features, count))), dim=1))
    network[0] = expanded


def remove_prediction_inputs(network, feature_dim, prediction_dim):
    """Retain [difference, query] columns, bias and RNG from the matched network."""
    first = network[0]
    first.weight = nn.Parameter(torch.cat((first.weight[:, :feature_dim].detach().clone(),
        first.weight[:, feature_dim+prediction_dim:].detach().clone()), dim=1))
    first.in_features -= prediction_dim


class QueryAwareClassificationAdapter(ClassificationNNCDHAdapter):
    adapter_design = "query_aware"
    uses_retrieval_input = True

    def __init__(self, feature_dim, num_classes, nominal_dim=0, hidden_dims=(64,32),
                 output_mode="nominal_residual_scores", dropout=0.):
        if nominal_dim:
            raise ValueError("query-aware experiments require fully represented numeric inputs")
        if output_mode != "nominal_residual_scores":
            raise ValueError("query-aware classification retains the additive score residual")
        super().__init__(feature_dim, num_classes, nominal_dim, hidden_dims, output_mode, dropout)
        append_zero_inputs(self.net, feature_dim)
        if not self.uses_retrieval_input:
            remove_prediction_inputs(self.net, feature_dim, num_classes)

    def forward(self, Delta_z_q, p0_q, Delta_u_q=None, *, query_features=None):
        if query_features is None or query_features.shape != Delta_z_q.shape:
            raise ValueError("the aligned query representation is required")
        if Delta_u_q is not None and Delta_u_q.shape[-1] != 0:
            raise ValueError("unrepresented nominal inputs are not supported")
        inputs = (Delta_z_q, p0_q, query_features) if self.uses_retrieval_input else (Delta_z_q, query_features)
        residual = self.net(torch.cat(inputs, dim=1)).tanh()
        return residual, p0_q+residual


class QueryAwareRegressionAdapter(NNCDHAdapter):
    adapter_design = "query_aware"
    uses_retrieval_input = True

    def __init__(self, feature_dim, label_dim=1, hidden_dims=(64,32)):
        super().__init__(feature_dim, label_dim, hidden_dims)
        append_zero_inputs(self.adapt_net_final.model, feature_dim)
        if not self.uses_retrieval_input:
            remove_prediction_inputs(self.adapt_net_final.model, feature_dim, label_dim)
        self.adapt_net_pair.requires_grad_(False)

    def build_final_features(self, query_features, aggregate_features, retrieval):
        if not self.uses_retrieval_input:
            return torch.cat((query_features-aggregate_features, query_features), dim=1)
        return torch.cat((query_features-aggregate_features, retrieval, query_features), dim=1)

    def forward_differences(self, difference, retrieval, query_features):
        if query_features.shape != difference.shape:
            raise ValueError("the aligned query representation is required")
        inputs = (difference, retrieval, query_features) if self.uses_retrieval_input else (difference, query_features)
        return retrieval+self.adapt_net_final(torch.cat(inputs, dim=1))


class QueryAwareNoPredictionClassificationAdapter(QueryAwareClassificationAdapter):
    adapter_design = "query_aware_no_prediction"
    uses_retrieval_input = False


class QueryAwareNoPredictionRegressionAdapter(QueryAwareRegressionAdapter):
    adapter_design = "query_aware_no_prediction"
    uses_retrieval_input = False


def query_adapter_type(task_type, design):
    if design not in QUERY_AWARE_DESIGNS or task_type not in {"classification", "regression"}:
        raise ValueError("unsupported query-aware adapter type")
    if task_type == "classification":
        return QueryAwareClassificationAdapter if design == "query_aware" else QueryAwareNoPredictionClassificationAdapter
    return QueryAwareRegressionAdapter if design == "query_aware" else QueryAwareNoPredictionRegressionAdapter
