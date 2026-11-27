"""T1 core compatibility and retrieval-event recording for stage adapters."""
from model.common.core import *
from model.common.core import retrieval_events as _core_retrieval_events
from .query_aware import is_query_aware


def retrieval_events(model, X, y, *, stream, run_id, top=5, adapter=None, query_nominal=None):
    """Preserve historical events and include stage-specific adaptation inputs.

    The stage adapter needs the full retrieval event, including learned distances;
    shared-core aggregate calls cannot provide those statistics.
    """
    from .outcomes import active_adapter, final_prediction, frozen_evaluation

    selected = active_adapter(model, adapter) if model.task_type == "classification" else (
        adapter if adapter is not None else getattr(model, "nn_cdh", None)
        if getattr(model, "adapt_enabled", True) else None)
    design = getattr(selected, "adapter_design", "residual")
    if design != "magnitude_direction" and not is_query_aware(selected):
        return _core_retrieval_events(model, X, y, stream=stream, run_id=run_id, top=top,
                                      adapter=adapter, query_nominal=query_nominal)
    device = model.cases.device
    out = []
    ids = model.active_case_ids()
    with frozen_evaluation(model, selected):
        for start in range(0, len(X), 256):
            xb = X[start:start + 256].to(device)
            r = model.retrieve(xb, exclude_identical=False)
            w = r["weights"]
            nominal = None if query_nominal is None else query_nominal[start:start + 256]
            diagnostics = {}
            pre, final, loss_kind = final_prediction(model, xb, r, adapter=selected,
                query_nominal=nominal, diagnostics=diagnostics)
            for b in range(len(xb)):
                sel = torch.nonzero(w[b] > 0).view(-1)
                event = {
                    "retrieval_event_id": f"{run_id}-{stream}-{start + b}",
                    "stream": stream, "query_index": start + b,
                    "case_ids": ids[r["case_indices"][sel]].tolist(),
                    "activations": w[b, sel].cpu().tolist(),
                    "distances": r["distances"][b, sel].cpu().tolist(),
                    "case_biases": model.biases[r["case_indices"][sel]].detach().cpu().tolist(),
                    "pre_adaptation_prediction": int(pre[b].argmax()) if model.task_type == "classification" else float(pre[b, 0]),
                    "final_prediction": int(final[b].argmax()) if model.task_type == "classification" else float(final[b].view(-1)[0]),
                    "final_prediction_loss_kind": loss_kind,
                    "target": int(y[start + b]) if model.task_type == "classification" else float(y[start + b]),
                }
                if model.task_type == "classification":
                    target = torch.nn.functional.one_hot(y[start + b].long(), pre.shape[1]).to(pre)
                    event.update(p0=pre[b].cpu().tolist(), residual_target=(target-pre[b]).cpu().tolist(),
                                 final_values=final[b].cpu().tolist())
                event.update(adapter_output_mode=diagnostics["output_mode"],
                    delta_z=diagnostics["delta_z"][b].cpu().tolist(),
                    delta_u=None if diagnostics["delta_u"] is None else diagnostics["delta_u"][b].cpu().tolist(),
                    predicted_residual=diagnostics["residual"][b].cpu().tolist(),
                    adapter_design=design)
                if is_query_aware(selected):
                    event["query_features"] = diagnostics["query_features"][b].cpu().tolist()
                else:
                    event.update(magnitude=float(diagnostics["magnitude"][b].item()),
                        raw_proposal=diagnostics["raw_proposal"][b].cpu().tolist(),
                        direction=diagnostics["direction"][b].cpu().tolist(),
                        neighborhood_statistics=diagnostics["neighborhood_statistics"][b].cpu().tolist())
                out.append(event)
    return out
