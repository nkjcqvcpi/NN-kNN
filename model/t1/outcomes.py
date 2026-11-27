"""T1 prediction routing; historical adapters delegate to the shared core.

Stage adapters explicitly route their extra inputs: neighborhood statistics for
magnitude/direction, or the query representation for the query-aware residual.
"""
from model.common.outcomes import *
from model.common.outcomes import final_prediction as _core_final_prediction

from .magnitude_direction import neighborhood_statistics
from .nominal import nominal_difference
from .query_aware import is_query_aware


def final_prediction(model, X, retrieval, *, adapter=None, case_mask=None,
                     query_case_ids=None, exclude_identical=False, query_nominal=None,
                     diagnostics=None):
    if model.task_type == "classification":
        selected = active_adapter(model, adapter)
    else:
        selected = adapter if adapter is not None else (
            getattr(model, "nn_cdh", None) if getattr(model, "adapt_enabled", True) else None)
    if is_query_aware(selected):
        w = retrieval["weights"]
        pre = w @ model.labels[retrieval["case_indices"]].float().view(w.shape[1], -1)
        if model.task_type == "classification":
            pre = pre/pre.sum(1, keepdim=True).clamp_min(1e-12)
        query = retrieval["query_features"]
        dz = query-w @ retrieval["case_features"]
        if model.task_type == "classification":
            residual, final = selected(dz, pre, query_features=query)
        else:
            final = selected.forward_differences(dz, pre, query)
            residual = final-pre
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(p0=pre.detach(), delta_z=dz.detach(), delta_u=None,
                               query_features=query.detach(), residual=residual.detach(),
                               final_values=final.detach(), adapter_design=selected.adapter_design,
                               output_mode=getattr(selected, "output_mode", "aggregate_regression_residual"))
        return pre, final, "cross_entropy" if model.task_type == "classification" else "squared_error"
    if getattr(selected, "adapter_design", "residual") != "magnitude_direction":
        return _core_final_prediction(model, X, retrieval, adapter=adapter,
            case_mask=case_mask, query_case_ids=query_case_ids, exclude_identical=exclude_identical,
            query_nominal=query_nominal, diagnostics=diagnostics)

    w = retrieval["weights"]
    labels = model.labels[retrieval["case_indices"]].float()
    statistics = neighborhood_statistics(retrieval, labels, task_type=model.task_type)
    pre = w @ labels.view(w.shape[1], -1)
    if model.task_type == "classification":
        pre = pre / pre.sum(1, keepdim=True).clamp_min(1e-12)
    dz = retrieval["query_features"] - w @ retrieval["case_features"]
    du = nominal_difference(selected, model, retrieval, query_nominal)
    parts = selected.forward_parts(dz, pre, statistics, du)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(p0=pre.detach(), delta_z=dz.detach(),
            delta_u=None if du is None else du.detach(), residual=parts.correction.detach(),
            raw_proposal=parts.raw_proposal.detach(), magnitude=parts.magnitude.detach(),
            direction=parts.direction.detach(), neighborhood_statistics=statistics.detach(),
            final_values=parts.final.detach(), adapter_design="magnitude_direction",
            output_mode=getattr(selected, "output_mode", "aggregate_regression_residual"))
    kind = "cross_entropy" if model.task_type == "classification" else "squared_error"
    return pre, parts.final, kind
