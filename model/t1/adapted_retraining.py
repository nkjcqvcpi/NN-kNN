"""Fixed-budget isolated continuations with a real final-output adapter.

This is the higher-cost PI comparison, not the efficient frozen removal audit.
Training examples and leave-one-out IDs stay fixed when memory is compacted.
Validation and test data are never accessed or used for checkpoint selection.
"""
import copy
import math
import time
from dataclasses import replace

import numpy as np
import torch

from .core import clone_optimizer
from .maintenance import CaseArchive, realign_optimizer_state
from .nominal import encoded_queries
from .reuse import _adapter_losses
from .sync import _forward_terms, frozen_parameters


def matched_random_continuation(model, core_optimizer, adapter, adapter_optimizer_state,
                                 data, core_cfg, reuse_cfg, store, score_cfg, retention_cfg,
                                 K, *, retrieval_epochs, adapter_epochs, lr_scale, seed):
    """Equal final capacity and update blocks, with protected random deletion.

This comparator respects protection/cohort floors but does not select or stop
using a prediction-loss budget. Report any loss-budget violation explicitly.
"""
    from .geometry import case_case_distance
    from .provenance import active_cohorts, score_cases
    from .retention import protection_requirements
    retention_cfg.validate()
    reuse_cfg.validate()
    if adapter is None or core_optimizer is None or adapter_optimizer_state is None:
        raise ValueError("Random continuation requires selected model/adapter states")
    if any(type(v) is not int or v < 0 for v in (retrieval_epochs, adapter_epochs)) or retrieval_epochs + adapter_epochs == 0:
        raise ValueError("Explicit continuation budgets must include at least one epoch")
    if not np.isfinite(lr_scale) or lr_scale <= 0:
        raise ValueError("Learning-rate scale must be finite and positive")
    if type(K) is not int or not 1 <= K <= model.case_count():
        raise ValueError("Random control requires an achievable final capacity")
    current, ad = copy.deepcopy(model), copy.deepcopy(adapter)
    opt = clone_optimizer(core_optimizer, current, core_cfg)
    ast = copy.deepcopy(adapter_optimizer_state)
    ids0 = current.active_case_ids().cpu().numpy()
    scores = score_cases(ids0, active_cohorts(current, data.reg_bins),
        current.biases[:current.case_count()].detach().cpu().numpy(), store, score_cfg)
    distances = case_case_distance(current).cpu().numpy() if retention_cfg.boundary_protect_fraction else None
    protected, floors, _ = protection_requirements(scores, retention_cfg, distances)
    required = sum(max(floors[c], int(protected[np.asarray(scores.cohorts) == c].sum())) for c in floors)
    if K < required:
        raise ValueError("Random control capacity violates protected floors")
    protected_ids = set(ids0[protected].tolist())
    rng = np.random.default_rng(seed)
    histories, removed = [], []
    while current.case_count() > K:
        ids = current.active_case_ids().cpu().numpy()
        cohorts = np.asarray(active_cohorts(current, data.reg_bins))
        eligible = [j for j, cid in enumerate(ids) if int(cid) not in protected_ids
                    and int((cohorts == cohorts[j]).sum()) > floors[cohorts[j]]]
        if not eligible:
            raise RuntimeError("Random control cannot reach matched capacity under protected floors")
        j = int(rng.choice(eligible))
        removed.append(int(ids[j]))
        current, opt, ad, aopt, info = continued_adapted_trial(current, opt, ad, ast, data, core_cfg, reuse_cfg,
            retrieval_epochs=retrieval_epochs, adapter_epochs=adapter_epochs, lr_scale=lr_scale, drop_slot=j)
        ast = copy.deepcopy(aopt.state_dict())
        histories.append(info)
    return current, opt, ad, ast, {
        "policy": "protected_random_deletion_then_continuation", "seed": seed,
        "n_before": len(ids0), "n_after": current.case_count(), "removed_case_ids": removed,
        "protected_case_ids": sorted(protected_ids), "cohort_floors": {str(k): v for k, v in floors.items()},
        "history": histories, "loss_budget_used_for_selection": False,
        "optimizer_updates": {phase: sum(h["optimizer_updates"][phase] for h in histories)
                              for phase in ("retrieval", "adapter")}}


def run_adapted_retrained_removal(model, core_optimizer, adapter, adapter_optimizer_state,
                                  data, core_cfg, reuse_cfg, reference, store, score_cfg,
                                  retention_cfg, K, *, retrieval_epochs, adapter_epochs,
                                  lr_scale, run_id):
    """Full-reference candidate search with cumulative loss and protected floors.

Each candidate and matched full-memory control starts from the same accepted
model, adapter, moments and RNG. A candidate's final prediction loss uses its
own continued adapter. Rejected forks cannot change the accepted memory.
"""
    from .geometry import case_case_distance
    from .provenance import active_cohorts, score_cases
    from .retention import protection_requirements
    from .retraining import reference_loss
    reference.validate()
    retention_cfg.validate()
    reuse_cfg.validate()
    if retention_cfg.policy != "removal_influence" or type(K) is not int or K < 1:
        raise ValueError("Adapted retrained removal needs full influence and positive integer K")
    # Reuse the continuation validation before returning even a no-removal run.
    if adapter is None or core_optimizer is None or adapter_optimizer_state is None:
        raise ValueError("Both selected optimizer states and the adapter are required")
    if model.task_type != "classification" or core_cfg.task_type != "classification" or adapter.output_mode != reuse_cfg.output_mode:
        raise ValueError("Classification and matching output semantics are required")
    if getattr(model, "classification_adapter", None) is not None or model.nn_cdh is not None:
        raise ValueError("Pass an external adapter with an unattached retrieval core")
    if len(data.y_train) == 0 or not torch.isin(model.active_case_ids().cpu(), torch.arange(len(data.y_train))).all():
        raise ValueError("Memory must bind to the fixed original training query IDs")
    if any(type(v) is not int or v < 0 for v in (retrieval_epochs, adapter_epochs)) or retrieval_epochs + adapter_epochs == 0:
        raise ValueError("Explicit continuation budgets must include at least one epoch")
    if not np.isfinite(lr_scale) or lr_scale <= 0:
        raise ValueError("Learning-rate scale must be finite and positive")
    current, ad = copy.deepcopy(model), copy.deepcopy(adapter)
    opt = clone_optimizer(core_optimizer, current, core_cfg)
    astate = copy.deepcopy(adapter_optimizer_state)
    ids0 = current.active_case_ids().cpu().numpy()
    scores = score_cases(ids0, active_cohorts(current, data.reg_bins),
                         current.biases[:current.case_count()].detach().cpu().numpy(), store, score_cfg)
    distances = case_case_distance(current).cpu().numpy() if retention_cfg.boundary_protect_fraction else None
    protected, floors, reasons = protection_requirements(scores, retention_cfg, distances)
    required = sum(max(floors[c], int(protected[np.asarray(scores.cohorts) == c].sum())) for c in floors)
    if min(K, len(ids0)) < required:
        raise ValueError("Capacity cannot satisfy protected cases and cohort floors")
    protected_ids = set(ids0[protected].tolist())
    def loss(m, a):
        return reference_loss(m, replace(reference, adapter=a))
    original_loss = loss(current, ad)
    if not np.isfinite(original_loss):
        raise ValueError("Nonfinite initial final-prediction reference loss")
    archive, trace, events = CaseArchive(), [], []
    totals = {name: {"retrieval": 0, "adapter": 0} for name in ("candidates", "controls", "accepted")}
    started = time.perf_counter()
    stop_reason = "capacity_reached"
    while current.case_count() > K:
        ids = current.active_case_ids().cpu().numpy()
        cohorts = np.asarray(active_cohorts(current, data.reg_bins))
        eligible = [j for j, cid in enumerate(ids) if int(cid) not in protected_ids
                    and int((cohorts == cohorts[j]).sum()) > floors[cohorts[j]]]
        if not eligible:
            stop_reason = "protection_limit"
            break
        before = loss(current, ad)
        def trial(slot=None):
            return continued_adapted_trial(current, opt, ad, astate, data, core_cfg, reuse_cfg,
                retrieval_epochs=retrieval_epochs, adapter_epochs=adapter_epochs,
                lr_scale=lr_scale, drop_slot=slot)
        control = trial()
        control_loss = loss(control[0], control[2])
        if not np.isfinite(control_loss):
            raise ValueError("Nonfinite matched control loss")
        for phase in totals["controls"]:
            totals["controls"][phase] += control[4]["optimizer_updates"][phase]
        candidates, best = [], None
        for j in eligible:
            candidate = trial(j)
            value = loss(candidate[0], candidate[2])
            if not np.isfinite(value):
                raise ValueError("Nonfinite candidate final-prediction reference loss")
            for phase in totals["candidates"]:
                totals["candidates"][phase] += candidate[4]["optimizer_updates"][phase]
            row = {"case_id": int(ids[j]), "loss_after_retraining": value,
                   "influence": value - before, "delta_vs_matched_training": value - control_loss,
                   "continuation": candidate[4]}
            candidates.append(row)
            if best is None or (value, int(ids[j])) < best[:2]:
                best = (value, int(ids[j]), j, candidate)
        value, cid, j, chosen = best
        cumulative = value - original_loss
        accepted = cumulative <= retention_cfg.allowed_loss_increase + 1e-10
        trace.append({"iteration": len(trace), "active_case_ids": ids.tolist(),
            "reference_denominator": len(reference.y), "reference_stream": reference.stream,
            "starting_reference_loss": before, "matched_full_memory_loss": control_loss,
            "control_continuation": control[4], "candidate_scores": candidates,
            "candidate_case_id": cid, "cumulative_loss_increase": cumulative,
            "accepted": accepted, "selection_loss": "final_prediction_only"})
        if not accepted:
            stop_reason = "cumulative_loss_budget"
            break
        initial_slot = int(np.flatnonzero(ids0 == cid)[0])
        event = {"maintenance_event_id": f"{run_id}-adapted-retrain-{len(events)}",
            "run_id": run_id, "case_id": cid, "action": "archive", "step": len(events),
            "reason_codes": reasons[initial_slot] + ["lowest_eligible_adapted_retrained_loss"],
            "reference_loss_after": value, "cumulative_loss_increase": cumulative,
            "capacity_before": len(ids), "capacity_after": len(ids) - 1,
            "policy_version": "adapted_removal_retrained/pi-20260920-v1",
            "restoration": {"archive_key": cid}}
        archive.add(current, j, event, len(events), event["reason_codes"])
        events.append(event)
        current, opt, ad, aopt, info = chosen
        astate = copy.deepcopy(aopt.state_dict())
        for phase in totals["accepted"]:
            totals["accepted"][phase] += info["optimizer_updates"][phase]
    return current, opt, ad, astate, archive, {
        "summary": {"policy": "adapted_removal_retrained", "n_before": len(ids0),
            "n_after": current.case_count(), "K": K, "capacity_reached": current.case_count() <= K,
            "stop_reason": stop_reason, "original_reference_loss": original_loss,
            "final_reference_loss": loss(current, ad),
            "allowed_loss_increase": retention_cfg.allowed_loss_increase,
            "protected_case_ids": sorted(protected_ids), "cohort_floors": {str(k): v for k, v in floors.items()},
            "optimizer_updates": totals, "selection_seconds": time.perf_counter() - started,
            "reference_stream": reference.stream, "selection_loss": "final_prediction_only",
            "compute_matching": "equal_examples_and_phase_budgets_not_equal_cost"},
        "events": events, "scoring_trace": trace}


def continued_adapted_trial(model, core_optimizer, adapter, adapter_optimizer_state,
                            data, core_cfg, reuse_cfg, *, retrieval_epochs,
                            adapter_epochs, lr_scale, drop_slot=None,
                            training_case_ids=None):
    """Return independent model/adapter/Adam forks at the fixed final epoch.

Retrieval updates minimize final classification loss through the frozen adapter;
adapter updates use the declared v0 residual/classification objective through
frozen retrieval. When both budgets are positive, alternate single epochs,
starting with retrieval. Equal budgets mean equal examples/updates, not cost.
"""
    reuse_cfg.validate()
    if model.task_type != "classification" or core_cfg.task_type != "classification":
        raise ValueError("This continuation package requires classification")
    if adapter is None or adapter_optimizer_state is None or core_optimizer is None:
        raise ValueError("Continuation requires both selected Adam states and an adapter")
    if getattr(model, "classification_adapter", None) is not None or model.nn_cdh is not None:
        raise ValueError("Pass an external adapter with an unattached retrieval core")
    if adapter.output_mode != reuse_cfg.output_mode:
        raise ValueError("Continuation output mode must match the selected adapter")
    if any(type(v) is not int or v < 0 for v in (retrieval_epochs, adapter_epochs)) or retrieval_epochs + adapter_epochs == 0:
        raise ValueError("Explicit nonnegative integer budgets must include at least one epoch")
    if not np.isfinite(lr_scale) or lr_scale <= 0:
        raise ValueError("Learning-rate scale must be finite and positive")
    if len(data.y_train) == 0 or len(data.X_train) != len(data.y_train):
        raise ValueError("A nonempty aligned training stream is required")
    qids = torch.arange(len(data.y_train)) if training_case_ids is None else torch.as_tensor(training_case_ids)
    if qids.ndim != 1 or len(qids) != len(data.y_train) or qids.dtype != torch.int64 or len(torch.unique(qids)) != len(qids):
        raise ValueError("Training query stable IDs must be unique aligned int64 values")
    if not torch.isin(model.active_case_ids().cpu(), qids.cpu()).all():
        raise ValueError("Active memory cases must bind to the declared training query IDs")
    if drop_slot is not None and (type(drop_slot) is not int or not 0 <= drop_slot < model.case_count()):
        raise ValueError("Removal slot must identify an active case")
    trial, ad = copy.deepcopy(model), copy.deepcopy(adapter)
    ropt = clone_optimizer(core_optimizer, trial, core_cfg)
    aopt = torch.optim.Adam([p for p in ad.parameters() if p.requires_grad], lr=reuse_cfg.lr)
    aopt.load_state_dict(copy.deepcopy(adapter_optimizer_state))
    if drop_slot is not None:
        n = trial.case_count()
        if n <= 1:
            raise ValueError("Cannot remove the only case")
        keep = np.delete(np.arange(n), drop_slot)
        trial.compact_cases(torch.as_tensor(keep))
        realign_optimizer_state(ropt, trial, keep, n)
    base_lr = {"case": core_cfg.lr_case, "glocal": core_cfg.lr_glocal, "feature": core_cfg.lr_feature}
    for group in ropt.param_groups:
        group["lr"] = base_lr[group["name"]] * lr_scale
    for group in aopt.param_groups:
        group["lr"] = reuse_cfg.lr * lr_scale
    device = trial.cases.device
    X, y, qids = data.X_train.to(device), data.y_train.to(device), qids.to(device)
    nominal = encoded_queries(ad, data, "train")
    if nominal is not None:
        nominal = nominal.to(device)
    phases = []
    for i in range(max(retrieval_epochs, adapter_epochs)):
        if i < retrieval_epochs:
            phases.append("retrieval")
        if i < adapter_epochs:
            phases.append("adapter")
    generator = torch.Generator().manual_seed(core_cfg.seed)
    history, updates = [], {"retrieval": 0, "adapter": 0}
    start = time.perf_counter()
    device_type = device.type if device.type in {"cuda", "xpu"} else "cuda"
    devices = [device.index or 0] if device.type in {"cuda", "xpu"} else []
    with torch.random.fork_rng(devices=devices, device_type=device_type):
        for epoch, phase in enumerate(phases, 1):
            perm = torch.randperm(len(y), generator=generator)
            loss_sum, post_sum = 0., 0.
            batch_count = 0
            for start_row in range(0, len(y), core_cfg.batch_size):
                b = perm[start_row:start_row + core_cfg.batch_size].to(device)
                ropt.zero_grad(set_to_none=True)
                aopt.zero_grad(set_to_none=True)
                query_nominal = None if nominal is None else nominal[b]
                if phase == "retrieval":
                    trial.train()
                    ad.eval()
                    with frozen_parameters(ad):
                        terms = _forward_terms(trial, ad, X[b], y[b], None,
                            output_mode=reuse_cfg.output_mode, near_scale=1.,
                            query_case_ids=qids[b], query_nominal=query_nominal)
                    loss = terms["L_post"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trial.parameters(), core_cfg.grad_clip)
                    ropt.step()
                    if core_cfg.mcb_enabled:
                        trial.update_momentum_encoder(core_cfg.mcb_momentum)
                else:
                    trial.eval()
                    ad.train()
                    with frozen_parameters(trial):
                        # Both residual modes use their declared objective. The
                        # neighborhood contract never supplies a raw embedding.
                        from .nominal import nominal_difference
                        r = trial.retrieve(X[b], exclude_identical=False, query_case_ids=qids[b])
                        w = r["weights"]
                        p0 = w @ trial.labels[r["case_indices"]].float()
                        p0 = p0 / p0.sum(1, keepdim=True).clamp_min(1e-12)
                        dz = r["query_features"] - w @ r["case_features"]
                        rh, final = ad(dz, p0, nominal_difference(ad, trial, r, query_nominal))
                        losses = _adapter_losses(ad, rh, final, p0, y[b], reuse_cfg)
                    loss = losses["loss"]
                    terms = {"L_post": losses["l_cls"]}
                    loss.backward()
                    aopt.step()
                if not torch.isfinite(loss.detach()):
                    raise ValueError("Nonfinite continuation loss")
                loss_sum += float(loss.detach()) * len(b)
                post_sum += float(terms["L_post"].detach()) * len(b)
                batch_count += 1
                updates[phase] += 1
            history.append({"epoch": epoch, "phase": phase, "examples": len(y),
                            "optimizer_updates": batch_count, "loss": loss_sum / len(y),
                            "final_classification_loss": post_sum / len(y)})
    trial.eval()
    ad.eval()
    expected_batches = math.ceil(len(y) / core_cfg.batch_size)
    assert updates == {"retrieval": retrieval_epochs * expected_batches,
                       "adapter": adapter_epochs * expected_batches}
    return trial, ropt, ad, aopt, {
        "history": history, "retrieval_epochs": retrieval_epochs,
        "adapter_epochs": adapter_epochs, "optimizer_updates": updates,
        "training_examples_per_epoch": len(y), "case_count": trial.case_count(),
        "retrieval_query_case_pairs": len(y) * trial.case_count() * len(phases),
        "seconds": time.perf_counter() - start,
        "selection": "fixed_final_epoch_no_validation_or_test",
        "objective_retrieval": "final_classification_loss",
        "objective_adapter": reuse_cfg.loss,
        "compute_matching": "equal_examples_and_phase_budgets_not_equal_cost"}
