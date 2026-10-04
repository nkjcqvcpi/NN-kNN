"""Final prediction semantics shared by provenance and maintenance scoring."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def frozen_evaluation(model, adapter=None):
    modules = list(model.modules())
    if adapter is not None:
        modules += [m for m in adapter.modules() if m not in modules]
    modes = [(m, m.training) for m in modules]
    try:
        for m, _ in modes:
            m.training = False
        with torch.no_grad():
            yield
    finally:
        for m, mode in modes:
            m.training = mode


def active_adapter(model, adapter=None):
    if adapter is not None:
        return adapter
    if getattr(model, "enable_classification_adapter", True):
        return getattr(model, "classification_adapter", None)
    return None


def final_prediction(model, X, retrieval, *, adapter=None, case_mask=None, query_case_ids=None, exclude_identical=False):
    """Use the actual reuse path; prediction losses exclude adaptation penalties."""
    w = retrieval["weights"]
    if not torch.isfinite(w).all() or not torch.allclose(w.sum(1), torch.ones(w.shape[0], device=w.device), atol=1e-5):
        raise ValueError("Maintenance needs a valid normalized neighborhood for every reference query")
    labels = model.labels[retrieval["case_indices"]].float()
    pre = w @ labels.view(w.shape[1], -1)
    if model.task_type == "classification":
        pre = pre / pre.sum(1, keepdim=True).clamp_min(1e-12)
        ad = active_adapter(model, adapter)
        if ad is None:
            return pre, pre, "nll"
        dz = retrieval["query_features"] - w @ retrieval["case_features"]
        _, final = ad(dz, pre, None)
        return pre, final, "cross_entropy" if ad.output_mode == "nominal_residual_scores" else "nll"
    if adapter is not None:
        final = adapter.forward_aggregate(retrieval["query_features"], retrieval["case_features"], labels, w)
    elif model.nn_cdh is not None and getattr(model, "adapt_enabled", True):
        final = model(X, exclude_identical=exclude_identical, case_mask=case_mask, query_case_ids=query_case_ids)[0]
    else:
        final = pre
    return pre, final, "squared_error"


def prediction_loss(prediction, y, loss_kind):
    if loss_kind == "cross_entropy":
        return F.cross_entropy(prediction, y.long().view(-1), reduction="none")
    if loss_kind == "nll":
        return F.nll_loss(prediction.clamp_min(1e-8).log(), y.long().view(-1), reduction="none")
    if loss_kind == "squared_error":
        return (prediction.view(-1) - y.float().view(-1)).square()
    raise ValueError(loss_kind)


def query_success(prediction, y, task_type, regression_success_tolerance=None):
    if task_type == "classification":
        return prediction.argmax(1) == y.long().view(-1)
    if regression_success_tolerance is None or not 0 <= regression_success_tolerance < float("inf"):
        raise ValueError("Regression outcome evidence requires an explicit finite error tolerance in target units")
    return (prediction.view(-1) - y.float().view(-1)).abs() <= regression_success_tolerance
