import pytest
from model.t3.utility import direct_set_audit


def cases():return [dict(case_id=8,content='left fact'),dict(case_id=9,content='right fact')]


def test_complementary_cases_and_fixed_order_survive_evaluator_mutation():
    original=cases();seen=[];journal=[]
    def evaluate(payload):
        ids=[c['case_id'] for c in payload];seen.append(ids)
        if payload:payload[0]['content']='changed by host'
        return dict(losses={'answer':float(len(ids)!=2)})
    audit=direct_set_audit(original,evaluate,outcome_sink=journal.append)
    assert seen==[[8,9],[9],[8],[]] and original==cases()
    assert [m['loss_differences']['answer'] for m in audit['marginal']]==[1,1]
    assert audit['set_loss_gain']=={'answer':1} and audit['pair_complementarity']=={'answer':1}
    assert journal==audit['outcomes']


def test_missing_full_feedback_is_unobserved_and_never_negative_credit():
    def evaluate(payload):
        return dict(losses=None,unobserved_reason='feedback not available') if len(payload)==2 else dict(losses={'answer':0})
    audit=direct_set_audit(cases(),evaluate)
    assert all(m['status']=='unobserved' and m['loss_differences'] is None for m in audit['marginal'])
    assert audit['set_loss_gain'] is None and audit['pair_complementarity'] is None


def test_harmful_case_and_single_case_empty_not_repeated():
    seen=[]
    def evaluate(payload):
        seen.append(len(payload));return dict(losses={'answer':float(bool(payload))})
    audit=direct_set_audit(cases()[:1],evaluate)
    assert seen==[1,0] and audit['marginal'][0]['loss_differences']=={'answer':-1}
    assert audit['pair_complementarity'] is None


def test_completed_task_failure_is_observed_and_not_dropped():
    def evaluate(payload):
        return dict(losses={'answer':float(len(payload)==2)},task_failure=len(payload)==2)
    audit=direct_set_audit(cases(),evaluate)
    assert audit['outcomes'][0]['outcome']['task_failure'] is True
    assert all(m['status']=='observed' and m['loss_differences']=={'answer':-1} for m in audit['marginal'])
    assert audit['set_loss_gain']=={'answer':-1}


def test_sink_failure_stops_before_next_host_and_objectives_cannot_drift():
    calls=[]
    def evaluate(payload):calls.append(payload);return dict(losses={'answer':0})
    def sink(_):raise RuntimeError('journal unavailable')
    with pytest.raises(RuntimeError,match='journal unavailable'):direct_set_audit(cases(),evaluate,outcome_sink=sink)
    assert len(calls)==1
    with pytest.raises(ValueError,match='dimensions'):
        direct_set_audit(cases(),lambda p:dict(losses={'answer' if len(p)==2 else 'self_rating':0}))
