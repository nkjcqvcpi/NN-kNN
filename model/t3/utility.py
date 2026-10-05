"""Direct no-refill outcome comparisons; no automatic feedback learning.

Evaluators receive only exact subsets of the recorded delivered set. Missing
outcomes remain unobserved. Loss differences are conditional on this
query/set/host,not a global reliability label or a retrieval relevance score.
"""
import copy
import hashlib
import json
import math


def direct_set_audit(evidence,evaluate,*,outcome_sink=None):
    """Evaluate full,each single removal and empty under one caller-bound host.

    evaluate(payload) returns a dict with `losses`: finite named lower-is-better
    objectives,or None plus a nonempty `unobserved_reason`. Other public outcome
    metadata is retained. The caller binds host/prompt/resource/scorer versions.
    A sink saves each actual outcome before the next evaluation. An unexpected
    evaluation/sink exception propagates; no retry,refill or fabricated outcome.
    """
    if not isinstance(evidence,(tuple,list)) or not evidence:
        raise ValueError('a nonempty recorded delivered set is required')
    sealed=copy.deepcopy(list(evidence));ids=[]
    for case in sealed:
        if not isinstance(case,dict) or type(case.get('case_id')) is not int or case['case_id']<0 or not isinstance(case.get('content'),str):
            raise ValueError('exact evidence requires stable IDs and original content')
        ids.append(case['case_id'])
    if len(set(ids))!=len(ids):raise ValueError('delivered set IDs must be unique')
    digest=hashlib.sha256(json.dumps(sealed,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    subsets=[('full',tuple(ids))]+[(f'remove:{i}',tuple(j for j in ids if j!=i)) for i in ids]
    if len(ids)>1:subsets.append(('empty',()))
    outcomes={};metric_names=None
    for label,selected in subsets:
        payload=[copy.deepcopy(case) for case in sealed if case['case_id'] in selected]
        outcome=copy.deepcopy(evaluate(payload))
        if not isinstance(outcome,dict):raise ValueError('an explicit objective outcome is required')
        losses=outcome.get('losses')
        if losses is None:
            if not isinstance(outcome.get('unobserved_reason'),str) or not outcome['unobserved_reason'].strip():
                raise ValueError('missing losses require an explicit unobserved reason')
        else:
            if not isinstance(losses,dict) or not losses or any(not isinstance(k,str) or not k or
                    isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for k,v in losses.items()):
                raise ValueError('finite named lower-is-better losses are required')
            if metric_names is None:metric_names=set(losses)
            elif set(losses)!=metric_names:raise ValueError('objective dimensions must remain fixed')
        record=dict(variant=label,case_ids=list(selected),outcome=outcome)
        outcomes[label]=record
        if outcome_sink is not None:outcome_sink(copy.deepcopy(record))
    empty=outcomes['empty' if len(ids)>1 else f'remove:{ids[0]}']['outcome']['losses']
    full=outcomes['full']['outcome']['losses']
    marginal=[]
    for case_id in ids:
        removed=outcomes[f'remove:{case_id}']['outcome']['losses']
        observed=full is not None and removed is not None
        marginal.append(dict(case_id=case_id,status='observed' if observed else 'unobserved',
            loss_differences={k:removed[k]-full[k] for k in full} if observed else None))
    gain={k:empty[k]-full[k] for k in full} if full is not None and empty is not None else None
    complementarity=None
    if len(ids)==2 and gain is not None:
        left=outcomes[f'remove:{ids[0]}']['outcome']['losses']
        right=outcomes[f'remove:{ids[1]}']['outcome']['losses']
        if left is not None and right is not None:
            complementarity={k:left[k]+right[k]-full[k]-empty[k] for k in full}
    return dict(delivered_case_ids=ids,delivered_payload_sha256=digest,outcomes=list(outcomes.values()),
        marginal=marginal,set_loss_gain=gain,pair_complementarity=complementarity,
        boundary='direct exact-set/no-refill outcomes; positive removal loss difference is helpful; missing is unobserved; no parameter update')
