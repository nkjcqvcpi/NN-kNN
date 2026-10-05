import copy

import pytest
import torch

from model.nnknn_rl_workflow import NNKNNPolicyNetwork, NNKNNValueNetwork
from model.t2.audit import audit_query, summarize_credit


def critic(top_k=2):
    m = NNKNNValueNetwork(2, case_capacity=3, top_k=top_k, use_glocal_weightor=False)
    m.add_cases(torch.tensor([[0.,0.],[0.,0.]]),torch.tensor([0.,2.]))
    return m


def test_critic_direct_removal_signs_and_same_event_forward_trace(monkeypatch):
    m=critic()
    calls=[]
    original=m.nnknn_model.retrieve
    def spy(*args,**kwargs):
        r=original(*args,**kwargs)
        calls.append(r)
        return r
    monkeypatch.setattr(m.nnknn_model,'retrieve',spy)
    event=audit_query(m,[0.,0.],stream='independent_mc',target=0.)
    assert len(calls)==3  # exactly one full retrieval and two interventions
    assert event['weights']==calls[0]['weights'][0].tolist()==[.5,.5]
    assert event['full_prediction']==1.
    byid={i['case_id']:i for i in event['interventions']}
    assert byid[0]['delta']==3.
    assert byid[1]['delta']==-1.
    assert byid[0]['C']==3. and byid[0]['H']==0.
    assert byid[1]['C']==0. and byid[1]['H']==1.


def test_top_one_removal_does_not_refill_from_nonretrieved_case():
    m=critic(top_k=1)
    with torch.no_grad(): m.nnknn_model.biases[:2].copy_(torch.tensor([0.,2.]))
    event=audit_query(m,[0.,0.],stream='training_gae',target=2.)
    assert len(event['interventions'])==1
    assert event['full_prediction']==2.
    assert event['interventions'][0]['removed_prediction']==0.


def test_audit_preserves_parameters_modes_and_rng_and_separates_streams():
    m=critic()
    m.train()
    m.nnknn_model.eval()
    modes=[sub.training for sub in m.modules()]
    before=copy.deepcopy(m.state_dict())
    rng=torch.random.get_rng_state().clone()
    events=[audit_query(m,[0.,0.],stream=s,target=t)
            for s,t in [('training_gae',0.),('independent_mc',2.)]]
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    assert torch.equal(rng,torch.random.get_rng_state())
    assert modes==[sub.training for sub in m.modules()]
    stats=summarize_credit(events)
    assert stats[('critic','training_gae')][0]['C']==3.
    assert stats[('critic','independent_mc')][0]['H']==1.


def test_actor_surrogate_requires_ready_policy_and_uses_advantage_sign():
    m=NNKNNPolicyNetwork(2,2,case_capacity=3,top_k=2,use_glocal_weightor=False)
    m.add_cases(torch.tensor([[0.,0.],[0.,0.]]),torch.tensor([0,1]))
    with pytest.raises(ValueError,match='warmup'):
        audit_query(m,[0.,0.],stream='actor_policy_surrogate',action=0,advantage=1.,policy_ready=False)
    pos=audit_query(m,[0.,0.],stream='actor_policy_surrogate',action=0,advantage=1.,policy_ready=True)
    neg=audit_query(m,[0.,0.],stream='actor_policy_surrogate',action=0,advantage=-1.,policy_ready=True)
    assert pos['full_prediction']==[.5,.5]
    for p,n in zip(pos['interventions'],neg['interventions']):
        assert p['delta']==pytest.approx(-n['delta'])
    assert pos['interventions'][0]['delta']>0.
    assert pos['interventions'][1]['delta']<0.


def test_undefined_target_or_sampling_cannot_silently_enter_credit():
    m=critic()
    with pytest.raises(ValueError,match='distinguish'):
        audit_query(m,[0.,0.],stream='value',target=0.)
    m.nnknn_model.sampling_cases_flag=True
    with pytest.raises(ValueError,match='deterministic'):
        audit_query(m,[0.,0.],stream='independent_mc',target=0.)


def test_uniform_behavior_mixture_has_no_case_surrogate_credit():
    m=NNKNNPolicyNetwork(2,2,case_capacity=3,top_k=2,use_glocal_weightor=False)
    m.add_cases(torch.tensor([[0.,0.],[0.,0.]]),torch.tensor([0,1]))
    e=audit_query(m,[0.,0.],stream='actor_policy_surrogate',action=0,advantage=2.,
                  policy_ready=True,behavior_epsilon=1.)
    assert all(i['delta']==0 and i['C']==0 and i['H']==0 for i in e['interventions'])
