import copy

import pytest
import torch

from model.nnknn_rl_workflow import NNKNNValueNetwork
from model.t2.audit import audit_query
from model.t2.quality import QualityLedger


def event(stream='training_gae',target=0.):
    m=NNKNNValueNetwork(2,case_capacity=3,top_k=2,use_glocal_weightor=False)
    m.add_cases(torch.zeros(2,2),torch.tensor([0.,2.]))
    e=audit_query(m,[0.,0.],stream=stream,target=target)
    e.update(snapshot_sha256='a'*64,phase='before_gradient',global_step=16)
    return e


def test_dedup_separate_targets_and_roundtrip_retired_rows():
    ledger=QualityLedger('run/seed8')
    gae=event()
    assert ledger.apply(gae,event_id='gae/1',active_ids=[0,1])
    assert not ledger.apply(gae,event_id='gae/1',active_ids=[0,1])
    mc=event('independent_mc',target=2.)
    assert ledger.apply(mc,event_id='mc/1',active_ids=[0,1])
    assert ledger.rows[('critic','training_gae',0)]['C']==3.
    assert ledger.rows[('critic','independent_mc',0)]['H']==1.
    assert ledger.rows[('critic','training_gae',0)]['exposure']==1
    restored=QualityLedger.from_state_dict(ledger.state_dict())
    assert restored.state_dict()==ledger.state_dict()
    assert set(restored.active_rows('critic','training_gae',[1]))=={1}
    assert ('critic','training_gae',0) in restored.rows # archived, never relabelled


@pytest.mark.parametrize('mutation',[
    lambda e:e['interventions'][-1].update(Q=2.),
    lambda e:e['interventions'][-1].update(delta=float('nan')),
    lambda e:e.update(case_ids=[0,0]),
    lambda e:e.update(stream='mixed_gae_mc'),
])
def test_invalid_transaction_never_partially_updates(mutation):
    ledger=QualityLedger('run')
    e=event();ledger.apply(e,event_id='valid',active_ids=[0,1])
    before=ledger.state_dict();bad=copy.deepcopy(e);mutation(bad)
    with pytest.raises(ValueError):ledger.apply(bad,event_id='bad',active_ids=[0,1])
    assert ledger.state_dict()==before


def test_changed_event_identity_and_inactive_credit_rejected():
    ledger=QualityLedger('run');e=event()
    ledger.apply(e,event_id='event',active_ids=[0,1])
    changed=copy.deepcopy(e);changed['global_step']=17
    with pytest.raises(ValueError,match='reused'):
        ledger.apply(changed,event_id='event',active_ids=[0,1])
    with pytest.raises(ValueError,match='inactive'):
        ledger.apply(e,event_id='other',active_ids=[1])
    stored=ledger.state_dict();stored['rows'][0]['Q']=0.
    with pytest.raises(ValueError,match='stored Q'):
        QualityLedger.from_state_dict(stored)


def test_finite_inputs_cannot_overflow_accumulated_support():
    ledger=QualityLedger('run')
    e=event()
    e['interventions'][0].update(delta=1e308,removed_loss=1e308,C=1e308,H=0.,Q=1.)
    ledger.apply(e,event_id='one',active_ids=[0,1])
    before=ledger.state_dict()
    with pytest.raises(ValueError,match='overflow'):
        ledger.apply(e,event_id='two',active_ids=[0,1])
    assert ledger.state_dict()==before
