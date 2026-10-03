"""Independent removal-plus-training candidates and matched extra-training controls.

The initial implementation deliberately supports retrieval-only supervised cores.
Each trial continues copied Adam moments for an explicit, fixed number of updates.
Selection uses only the designated maintenance reference, never test/validation.
"""

import copy
import time
from dataclasses import replace

import numpy as np
import torch

from .core import clone_optimizer, train_retrieval
from .geometry import case_case_distance
from .maintenance import CaseArchive, realign_optimizer_state
from .outcomes import active_adapter, final_prediction, frozen_evaluation, prediction_loss
from .provenance import active_cohorts, score_cases
from .retention import protection_requirements


@torch.no_grad()
def reference_loss(model, reference):
    reference.validate()
    device = model.cases.device
    total = 0.0
    with frozen_evaluation(model, reference.adapter):
        for start in range(0, len(reference.y), 128):
            X = reference.X[start:start + 128].to(device)
            y = reference.y[start:start + 128].to(device)
            ids = None if reference.query_case_ids is None else reference.query_case_ids[start:start + 128].to(device)
            r = model.retrieve(X, exclude_identical=False, query_case_ids=ids)
            _, p, kind = final_prediction(model, X, r, adapter=reference.adapter, query_case_ids=ids)
            total += float(prediction_loss(p, y, kind).sum())
    return total / len(reference.y)


def continued_trial(model, optimizer, data, core_cfg, *, epochs, lr_scale, drop_slot=None):
    """A fresh state fork; identical RNG and minibatches for every candidate/control."""
    if epochs < 1 or not np.isfinite(lr_scale) or lr_scale <= 0:
        raise ValueError("Retraining needs a positive fixed epoch budget and learning-rate scale")
    trial = copy.deepcopy(model)
    opt = clone_optimizer(optimizer, trial, core_cfg)
    if drop_slot is not None:
        n = trial.case_count()
        keep = np.delete(np.arange(n), drop_slot)
        if len(keep) == 0:
            raise ValueError("Cannot remove the only case")
        trial.compact_cases(torch.as_tensor(keep))
        realign_optimizer_state(opt, trial, keep, n)
    cfg = replace(core_cfg, patience=epochs)
    # Dropout and any device RNG are restored after the trial; each fork starts
    # from the same RNG state. train_retrieval uses a seed-local permutation.
    devices = [trial.cases.device.index] if trial.cases.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        result = train_retrieval(trial, data.X_train, data.y_train, data.X_val, data.y_val,
                                 cfg, epochs=epochs, optimizer=opt, lr_scale=lr_scale,
                                 select_best=False)
    if result.epochs_run != epochs:
        raise RuntimeError("Candidate/control did not execute the fixed training budget")
    return trial, opt, result


def run_retrained_removal(base, optimizer, data, core_cfg, reference, store, score_cfg,
                          retention_cfg, K, *, epochs, lr_scale, run_id):
    """Sequential independent trials, accepting only within the original-loss budget.

Reports both PI influence (trial minus starting loss) and the difference from
unmodified memory trained for the same budget. Every round refreshes all trials.
The accepted model continues the selected trial; rejected trials never alter it.
"""
    reference.validate()
    retention_cfg.validate()
    if active_adapter(base, reference.adapter) is not None or base.nn_cdh is not None:
        raise ValueError("Retrained removal currently supports retrieval-only cores")
    if retention_cfg.policy != "removal_influence":
        raise ValueError("Retrained removal uses full-query removal influence")
    if K < 1 or epochs < 1:
        raise ValueError("Capacity and retraining epochs must be positive")
    model = copy.deepcopy(base)
    opt = clone_optimizer(optimizer, model, core_cfg)
    ids0 = model.active_case_ids().cpu().numpy()
    scores = score_cases(ids0, active_cohorts(model, data.reg_bins),
                         model.biases[:model.case_count()].detach().cpu().numpy(), store, score_cfg)
    distance = case_case_distance(model).cpu().numpy() if retention_cfg.boundary_protect_fraction else None
    protected, floors, reasons = protection_requirements(scores, retention_cfg, distance)
    required = sum(max(floors[c], int(protected[np.asarray(scores.cohorts) == c].sum())) for c in floors)
    if min(K, len(ids0)) < required:
        raise ValueError("Capacity cannot satisfy protected cases and cohort floors")
    protected_ids = set(ids0[protected].tolist())
    original_loss = reference_loss(model, reference)
    archive, trace, events = CaseArchive(), [], []
    stop_reason = "capacity_reached"
    trial_epochs = accepted_epochs = control_epochs = 0
    start = time.perf_counter()
    while model.case_count() > K:
        ids = model.active_case_ids().cpu().numpy()
        cohorts = np.asarray(active_cohorts(model, data.reg_bins))
        eligible = [j for j, cid in enumerate(ids) if int(cid) not in protected_ids
                    and int((cohorts == cohorts[j]).sum()) > floors[cohorts[j]]]
        if not eligible:
            stop_reason = "protection_limit"
            break
        before = reference_loss(model, reference)
        control, _, ctr = continued_trial(model, opt, data, core_cfg, epochs=epochs, lr_scale=lr_scale)
        control_loss = reference_loss(control, reference)
        control_epochs += ctr.epochs_run
        candidates, best = [], None
        for j in eligible:
            trial, trial_opt, trained = continued_trial(model, opt, data, core_cfg,
                                                       epochs=epochs, lr_scale=lr_scale, drop_slot=j)
            loss = reference_loss(trial, reference)
            if not np.isfinite(loss):
                raise ValueError("Nonfinite retraining candidate loss")
            trial_epochs += trained.epochs_run
            row = {"case_id": int(ids[j]), "loss_after_retraining": loss,
                   "influence": loss - before, "delta_vs_matched_training": loss - control_loss,
                   "epochs_run": trained.epochs_run, "history": trained.history}
            candidates.append(row)
            if best is None or (loss, int(ids[j])) < (best[0], best[1]):
                best = (loss, int(ids[j]), j, trial, trial_opt)
        loss, cid, j, chosen, chosen_opt = best
        cumulative = loss - original_loss
        accepted = cumulative <= retention_cfg.allowed_loss_increase + 1e-10
        trace.append({"iteration": len(trace), "active_case_ids": ids.tolist(),
                      "reference_denominator": len(reference.y), "starting_reference_loss": before,
                      "matched_full_memory_loss": control_loss, "control_history": ctr.history,
                      "candidate_scores": candidates, "candidate_case_id": cid,
                      "cumulative_loss_increase": cumulative, "accepted": accepted,
                      "reference_stream": reference.stream, "fixed_epochs_per_trial": epochs})
        if not accepted:
            stop_reason = "cumulative_loss_budget"
            break
        initial_slot = int(np.flatnonzero(ids0 == cid)[0])
        event = {"maintenance_event_id": f"{run_id}-retrain-{len(events)}", "run_id": run_id,
                 "case_id": cid, "action": "archive", "step": len(events),
                 "reason_codes": reasons[initial_slot] + ["lowest_eligible_retrained_loss"],
                 "cumulative_loss_increase": cumulative, "reference_loss_after": loss,
                 "capacity_before": len(ids), "capacity_after": len(ids) - 1,
                 "policy_version": "removal_retrained/pi-20260920-v1", "restoration": {"archive_key": cid}}
        archive.add(model, j, event, len(events), event["reason_codes"])
        events.append(event)
        model, opt = chosen, chosen_opt
        accepted_epochs += epochs
    summary = {"policy": "removal_retrained", "n_before": len(ids0), "K": K,
               "n_after": model.case_count(), "capacity_reached": model.case_count() <= K,
               "stop_reason": stop_reason, "original_reference_loss": original_loss,
               "final_reference_loss": reference_loss(model, reference),
               "allowed_loss_increase": retention_cfg.allowed_loss_increase,
               "accepted_model_training_epochs": accepted_epochs,
               "candidate_training_epochs_total": trial_epochs,
               "per_round_control_training_epochs_total": control_epochs,
               "selection_seconds": time.perf_counter() - start,
               "candidate_isolation": "copied_model_and_adam_same_rng",
               "checkpoint_selection": "fixed_final_epoch_no_validation_selection"}
    return model, opt, archive, {"summary": summary, "events": events, "scoring_trace": trace}
