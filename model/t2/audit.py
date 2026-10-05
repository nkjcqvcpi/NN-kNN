"""Direct fixed-event-set removal, separate actor surrogate and critic targets."""
from contextlib import contextmanager
import math

import torch

from model.nnknn_rl_workflow import NNKNNPolicyNetwork, NNKNNValueNetwork


@contextmanager
def evaluation(module):
    modes = [(m, m.training) for m in module.modules()]
    module.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for m, mode in modes:
            m.training = mode


def policy_output(values):
    values = values.clamp_min(0)
    total = values.sum(dim=1, keepdim=True)
    return torch.where(total > 0, values / total.clamp_min(1e-12),
                       torch.full_like(values, 1 / values.shape[1]))


def audit_query(model, query, *, stream, target=None, action=None,
                advantage=None, policy_ready=None, smoothing=1.0):
    """Audit an evaluation query without changing memories, parameters or RNG.

    Critic Delta is removal squared loss minus full squared loss. Actor Delta
    uses -advantage*log(pi(action)), a declared surrogate, never return utility.
    Remove only from the actual positive-weight event set with no refill.
    """
    critic = isinstance(model, NNKNNValueNetwork)
    actor = isinstance(model, NNKNNPolicyNetwork)
    if not (critic or actor):
        raise TypeError('audit requires a standalone NN-kNN actor or critic')
    if critic and stream not in {'training_gae', 'independent_mc'}:
        raise ValueError('critic stream must distinguish GAE training from independent MC')
    if actor and stream != 'actor_policy_surrogate':
        raise ValueError('actor surrogate stream must be explicit')
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError('smoothing must be finite and positive')
    core = model.nnknn_model
    if core.nn_cdh is not None or getattr(core, 'classification_adapter', None) is not None:
        raise ValueError('adapter-aware RL auditing is not yet defined')
    if core.sampling_cases_flag:
        raise ValueError('fixed-event audits require deterministic full-bank retrieval')
    q = torch.as_tensor(query, dtype=core.cases.dtype, device=core.cases.device).reshape(1, -1)
    if critic:
        if target is None or not math.isfinite(float(target)):
            raise ValueError('critic requires a finite declared target')
    else:
        if policy_ready is not True:
            raise ValueError('warmup/uniform actions have no case-attributable surrogate')
        if action is None or not 0 <= int(action) < model.action_dim or advantage is None or not math.isfinite(float(advantage)):
            raise ValueError('actor requires an executed action and finite signed advantage')
    if model.case_entries <= 0:
        raise ValueError('empty memory has no retrieved cases to audit')
    def prediction(output):
        return output.reshape(-1)[0] if critic else policy_output(output)[0]
    def loss(value):
        return (value - float(target)).square() if critic else -float(advantage) * value[int(action)].clamp_min(1e-12).log()
    with evaluation(model):
        outputs = core(q, return_retrieval=True)
        r = outputs[-1]
        full = prediction(outputs[0])
        full_loss = loss(full)
        indices, weights = r['case_indices'], r['weights'][0]
        selected = indices[weights > 0]
        base_mask = torch.zeros(model.case_entries, dtype=torch.bool, device=q.device)
        base_mask[selected] = True
        events = []
        for slot in selected.tolist():
            mask = base_mask.clone()
            mask[slot] = False
            removed = prediction(core(q, case_mask=mask)[0])
            removed_loss = loss(removed)
            delta = float((removed_loss - full_loss).cpu())
            c, h = max(delta, 0.), max(-delta, 0.)
            events.append(dict(case_id=int(core.case_ids[slot]), removed_prediction=removed.cpu().tolist(),
                removed_loss=float(removed_loss.cpu()), delta=delta, C=c, H=h,
                Q=(c + smoothing) / (c + h + 2*smoothing)))
        return dict(role='critic' if critic else 'actor', stream=stream,
            target=float(target) if critic else None, action=int(action) if actor else None,
            advantage=float(advantage) if actor else None, query=q[0].cpu().tolist(),
            full_prediction=full.cpu().tolist(), full_loss=float(full_loss.cpu()),
            case_ids=core.case_ids[indices].cpu().tolist(), weights=weights.cpu().tolist(),
            distances=r['distances'][0].cpu().tolist(), biases=core.biases[indices].cpu().tolist(),
            feature_distance_contributions=r['feature_distance_contributions'][0].cpu().tolist(),
            cases=core.cases[indices].cpu().tolist(), labels=core.labels[indices].cpu().tolist(),
            interventions=events, removal_rule='fixed_actual_positive_weight_set_no_refill',
            empty_fallback='zero_value' if critic else 'uniform_policy',
            credit_rule='unweighted_positive_negative_removal_loss_delta', smoothing=smoothing)


def summarize_credit(events):
    """Never combine roles or training/MC streams into one Q statistic."""
    result = {}
    for event in events:
        key = (event['role'], event['stream'])
        group = result.setdefault(key, {})
        for intervention in event['interventions']:
            case_id = intervention['case_id']
            row = group.setdefault(case_id, dict(C=0., H=0., exposure=0, smoothing=event['smoothing']))
            if row['smoothing'] != event['smoothing']:
                raise ValueError('cannot aggregate different smoothing priors')
            row['C'] += intervention['C']
            row['H'] += intervention['H']
            row['exposure'] += 1
            row['Q'] = (row['C']+row['smoothing'])/(row['C']+row['H']+2*row['smoothing'])
    return result
