"""September 20 candidate measures; no selected general maintenance formula.

Caches belong to one scoring call (one frozen checkpoint and active case base).
Rebuilding after every actual removal prevents stale-cache selection.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .outcomes import final_prediction, frozen_evaluation, prediction_loss, query_success


@dataclass
class MaintenanceReference:
    X: torch.Tensor
    y: torch.Tensor
    query_case_ids: torch.Tensor | None = None
    regression_success_tolerance: float | None = None
    adapter: object = None
    stream: str = "train_loo"
    query_nominal: torch.Tensor | None = None

    def validate(self):
        if self.stream not in {"train_loo", "maintenance_split"}:
            raise ValueError("Maintenance selection cannot use test or validation queries")
        if self.X.shape[0] == 0 or self.X.shape[0] != self.y.shape[0]:
            raise ValueError("A nonempty aligned reference set is required")
        if self.stream == "train_loo" and self.query_case_ids is None:
            raise ValueError("Training reference queries require stable IDs for leave-one-out")
        if self.query_case_ids is not None and self.query_case_ids.numel() != len(self.y):
            raise ValueError("Reference IDs must align with queries")
        if self.query_nominal is not None and self.query_nominal.shape[0] != len(self.y):
            raise ValueError("Reference nominal values must align with queries")


def _retrieve(model, X, qids, mask=None):
    return model.retrieve(X, exclude_identical=False, case_mask=mask, query_case_ids=qids)


@torch.no_grad()
def removal_influence(model, reference, *, cached_threshold=None, validate_cache=True, batch_size=128):
    """Mean final-loss difference, with the full |D| denominator in both modes.

Mask candidates one at a time without changing learned parameters. Thresholded
cache scores are approximations; full scores are computed for validation too.
"""
    reference.validate()
    if cached_threshold is not None and not 0 <= cached_threshold <= 1:
        raise ValueError("Cache activation threshold must be in [0, 1]")
    n, nq = model.case_count(), len(reference.y)
    full, cached, served = np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
    cache_query_ids = [[] for _ in range(n)]
    rerun_queries = validation_queries = 0
    baseline_loss = 0.0
    device = model.cases.device
    with frozen_evaluation(model, reference.adapter):
        for start in range(0, nq, batch_size):
            X = reference.X[start:start + batch_size].to(device)
            y = reference.y[start:start + batch_size].to(device)
            ids = None if reference.query_case_ids is None else reference.query_case_ids[start:start + batch_size].to(device)
            nominal = None if reference.query_nominal is None else reference.query_nominal[start:start + batch_size]
            r = _retrieve(model, X, ids)
            _, p, loss_kind = final_prediction(model, X, r, adapter=reference.adapter, query_case_ids=ids, query_nominal=nominal)
            baseline = prediction_loss(p, y, loss_kind)
            baseline_loss += float(baseline.sum())
            for i in range(n):
                mask = torch.ones(n, dtype=torch.bool, device=device)
                mask[i] = False
                selected = r["weights"][:, i] > (0.0 if cached_threshold is None else cached_threshold)
                served[i] += int(selected.sum())
                cache_query_ids[i].extend((torch.nonzero(selected).view(-1).cpu() + start).tolist())
                if cached_threshold is None:
                    rw = _retrieve(model, X, ids, mask)
                    _, pw, kind = final_prediction(model, X, rw, adapter=reference.adapter, case_mask=mask, query_case_ids=ids, query_nominal=nominal)
                    delta = prediction_loss(pw, y, kind) - baseline
                    full[i] += float(delta.sum())
                    cached[i] += float(delta[selected].sum())
                    rerun_queries += len(y)
                else:
                    if selected.any():
                        subids = None if ids is None else ids[selected]
                        rw = _retrieve(model, X[selected], subids, mask)
                        subnominal = None if nominal is None else nominal.to(device)[selected]
                        _, pw, kind = final_prediction(model, X[selected], rw, adapter=reference.adapter, case_mask=mask, query_case_ids=subids, query_nominal=subnominal)
                        cached[i] += float((prediction_loss(pw, y[selected], kind) - baseline[selected]).sum())
                        rerun_queries += int(selected.sum())
                    if validate_cache:
                        rw = _retrieve(model, X, ids, mask)
                        _, pw, kind = final_prediction(model, X, rw, adapter=reference.adapter, case_mask=mask, query_case_ids=ids, query_nominal=nominal)
                        full[i] += float((prediction_loss(pw, y, kind) - baseline).sum())
                        validation_queries += len(y)
    full /= nq
    cached /= nq
    has_full = cached_threshold is None or validate_cache
    return {"full": full if has_full else None, "cached": cached, "cache_query_counts": served,
            "cache_query_ids": cache_query_ids, "candidate_rerun_queries": rerun_queries,
            "validation_rerun_queries": validation_queries,
            "baseline_loss": baseline_loss / nq, "n_reference_queries": nq,
            "cache_max_absolute_error": float(np.max(np.abs(full - cached))) if has_full else None,
            "cache_validation": "full_reference_set" if has_full else "not_validated", "prediction_loss_only": True}


@torch.no_grad()
def coverage_reachability(model, reference, *, activation_threshold, zero_reachability, batch_size=128):
    reference.validate()
    if not 0 <= activation_threshold <= 1:
        raise ValueError("Activation threshold must be in [0, 1]")
    if zero_reachability not in {"error", "zero", "infinite"}:
        raise ValueError("Choose an explicit zero-reachability policy: error, zero, or infinite")
    n, nq = model.case_count(), len(reference.y)
    cover = np.zeros(n)
    device = model.cases.device
    with frozen_evaluation(model, reference.adapter):
        for start in range(0, nq, batch_size):
            X = reference.X[start:start + batch_size].to(device)
            y = reference.y[start:start + batch_size].to(device)
            ids = None if reference.query_case_ids is None else reference.query_case_ids[start:start + batch_size].to(device)
            nominal = None if reference.query_nominal is None else reference.query_nominal[start:start + batch_size]
            r = _retrieve(model, X, ids)
            _, p, _ = final_prediction(model, X, r, adapter=reference.adapter, query_case_ids=ids, query_nominal=nominal)
            success = query_success(p, y, model.task_type, reference.regression_success_tolerance)
            cover += ((r["weights"] > activation_threshold) & success[:, None]).sum(0).cpu().numpy()
        # Evaluate every active case problem separately, with stable-ID self exclusion.
        reach = np.zeros(n)
        for start in range(0, n, batch_size):
            X = model.cases[start:start + batch_size]
            ids = model.active_case_ids()[start:start + batch_size]
            labels = model.labels[start:start + batch_size]
            y = labels.argmax(1) if model.task_type == "classification" else labels.view(-1)
            r = _retrieve(model, X, ids)
            nominal = (reference.adapter.values_for_ids(ids) if getattr(reference.adapter, "nominal_dim", 0) else None)
            _, p, _ = final_prediction(model, X, r, adapter=reference.adapter, query_case_ids=ids, query_nominal=nominal)
            success = query_success(p, y, model.task_type, reference.regression_success_tolerance)
            reach[start:start + batch_size] = ((r["weights"] > activation_threshold) & success[:, None]).sum(1).cpu().numpy()
    zeros = reach == 0
    if zeros.any() and zero_reachability == "error":
        raise ValueError("Zero reachability encountered; no implicit epsilon or protection is allowed")
    ratio = np.divide(cover, reach, out=np.zeros(n), where=~zeros)
    if zero_reachability == "infinite":
        ratio[zeros] = np.inf
    return {"coverage": cover, "reachability": reach, "coverage_reachability": ratio,
            "zero_reachability_count": int(zeros.sum()), "n_reference_queries": nq}
