"""Exploratory fresh-reference Q selection with an absolute loss guard."""
from dataclasses import dataclass
import math

import torch

from model.nnknn_rl_workflow import NNKNNPolicyNetwork, NNKNNValueNetwork
from model.t2.audit import evaluation, policy_output


@dataclass
class TrainingReference:
    queries: torch.Tensor
    stream: str
    targets: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    behavior_epsilons: torch.Tensor | None = None
    policy_ready: torch.Tensor | None = None
    probability_floor: float = 1e-8


def constrained_retain(model,reference,*,keep_capacity,loss_budget=0.,min_exposure=1,
                       protected_ids=(),smoothing=1.):
    """Fresh direct full-bank removal trials; refill allowed, no gradients.

    Unknown cases relative to this original reference are protected. Rank only
    loss-budget-feasible removals by lowest fresh Q, then loss and stable ID.
    This is an exploratory selector, not an approved retention theorem.
    """
    critic=isinstance(model,NNKNNValueNetwork)
    actor=isinstance(model,NNKNNPolicyNetwork)
    if not (actor or critic):raise TypeError('standalone NN-kNN role required')
    n=model.case_entries
    if type(keep_capacity) is not int or not 1<=keep_capacity<=n:raise ValueError('invalid target capacity')
    if not math.isfinite(loss_budget) or loss_budget<0 or not math.isfinite(smoothing) or smoothing<=0:
        raise ValueError('invalid loss budget or prior')
    if type(min_exposure) is not int or min_exposure<1:raise ValueError('positive min_exposure required')
    if reference.stream!=('training_gae' if critic else 'actor_policy_surrogate'):
        raise ValueError('only explicit training references may select retention; MC is diagnostic')
    core=model.nnknn_model
    if core.nn_cdh is not None or getattr(core,'classification_adapter',None) is not None or core.sampling_cases_flag:
        raise ValueError('adapter/sampling retention semantics undefined')
    device=core.cases.device
    queries=torch.as_tensor(reference.queries,dtype=core.cases.dtype,device=device)
    if queries.ndim!=2 or queries.shape[1]!=model.obs_dim or not len(queries) or not bool(torch.isfinite(queries).all()):raise ValueError('finite nonempty query matrix required')
    ids=core.active_case_ids().tolist()
    protected=set(protected_ids)
    if any(type(i) is not int or i<0 for i in protected):raise ValueError('invalid protected identity')
    if not protected<=set(ids):raise ValueError('protected identity is inactive')
    if critic:
        targets=torch.as_tensor(reference.targets,dtype=queries.dtype,device=device).reshape(-1)
        if len(targets)!=len(queries) or not bool(torch.isfinite(targets).all()):raise ValueError('invalid GAE targets')
    else:
        if reference.policy_ready is None:raise ValueError('actor requires actual ready-policy provenance')
        ready=torch.as_tensor(reference.policy_ready,device=device).reshape(-1)
        if ready.dtype!=torch.bool or len(ready)!=len(queries) or not bool(ready.all()):
            raise ValueError('warmup actions cannot select case retention')
        if reference.actions is None or reference.advantages is None or reference.behavior_epsilons is None:
            raise ValueError('actor needs actual actions, signed advantages and epsilon')
        actions=torch.as_tensor(reference.actions,device=device).reshape(-1)
        advantages=torch.as_tensor(reference.advantages,dtype=queries.dtype,device=device).reshape(-1)
        eps=torch.as_tensor(reference.behavior_epsilons,dtype=queries.dtype,device=device).reshape(-1)
        if any(len(v)!=len(queries) for v in (actions,advantages,eps)) or not bool(torch.isfinite(advantages).all()) or not bool(torch.isfinite(eps).all()):
            raise ValueError('invalid actor reference')
        if bool(((actions<0)|(actions>=model.action_dim)|(actions!=actions.long())).any()) or bool(((eps<0)|(eps>1)).any()):raise ValueError('invalid action or epsilon')
        actions=actions.long()
        if not math.isfinite(reference.probability_floor) or reference.probability_floor<=0:raise ValueError('invalid probability floor')
    def forward(mask):
        output=core(queries,case_mask=mask,return_retrieval=True)
        if critic:losses=(output[0].flatten()-targets).square()
        else:
            probs=policy_output(output[0]);behavior=(1-eps[:,None])*probs+eps[:,None]/model.action_dim
            losses=-advantages*behavior.gather(1,actions[:,None]).flatten().clamp_min(reference.probability_floor).log()
        if not bool(torch.isfinite(losses).all()):raise ValueError('nonfinite retention loss')
        return losses,output[-1]
    with evaluation(model):
        mask=torch.ones(n,dtype=torch.bool,device=device)
        original,r=forward(mask)
        exposure=torch.zeros(n,dtype=torch.long,device=device)
        exposure[r['case_indices']]=(r['weights']>0).sum(0)
        eligible={i for i in range(n) if int(exposure[i])>=min_exposure and ids[i] not in protected}
        initial=float(original.mean())
        current=original
        trials=[];accepted=[]
        labels=model.action_tensor().tolist() if actor else None
        while int(mask.sum())>keep_capacity:
            round_trials=[]
            for slot in sorted(eligible,key=lambda i:ids[i]):
                if not mask[slot]:continue
                if actor and sum(bool(mask[i]) and labels[i]==labels[slot] for i in range(n))<=model.min_cases_per_action:continue
                candidate_mask=mask.clone();candidate_mask[slot]=False
                losses,_=forward(candidate_mask)
                delta=losses-current
                if not bool(torch.isfinite(delta).all()):raise ValueError('nonfinite removal delta')
                c=float(delta.clamp_min(0).sum());h=float((-delta).clamp_min(0).sum())
                if not math.isfinite(c+h+2*smoothing):raise ValueError('nonfinite support mass')
                loss=float(losses.mean());q=(c+smoothing)/(c+h+2*smoothing)
                record=dict(round=len(accepted),case_id=ids[slot],mean_loss=loss,query_losses=losses.cpu().tolist(),
                    C=c,H=h,Q=q,within_budget=loss<=initial+loss_budget+1e-7)
                trials.append(record)
                if record['within_budget']:round_trials.append((q,loss,ids[slot],slot,losses))
            if not round_trials:break
            _,_,case_id,slot,current=min(round_trials,key=lambda t:t[:3])
            mask[slot]=False;accepted.append(case_id)
        keep=mask.nonzero().flatten()
        final_ids=[ids[i] for i in keep.tolist()]
        if actor:removed=core.compact_cases(keep)
        else:removed=model._compact_cases(keep)
        return dict(stream=reference.stream,reference_queries=len(queries),original_loss=initial,
            final_loss=float(current.mean()),loss_budget=loss_budget,initial_query_losses=original.cpu().tolist(),
            final_query_losses=current.cpu().tolist(),initial_ids=ids,final_ids=final_ids,
            exposure=exposure.cpu().tolist(),protected_ids=sorted(protected),accepted=accepted,trials=trials,
            target_capacity=keep_capacity,actual_capacity=model.case_entries,removed=removed,
            reached=model.case_entries==keep_capacity,selector='fresh_Q_then_loss_then_ID_with_absolute_original_loss_guard',
            retrieval_rule='full_current_bank_retrieval_with_refill',unknown_policy='protect_original_reference_unexposed')
