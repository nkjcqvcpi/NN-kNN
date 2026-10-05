import copy

import pytest
import torch

from model.nnknn_rl_workflow import NNKNNValueNetwork,NNKNNPolicyNetwork,_realign_optimizer_case_state,_build_nnknn_target_value_model,_align_nnknn_target_case_store
from model.t2.retention import TrainingReference,constrained_retain


def critic(top_k=3):
    m=NNKNNValueNetwork(2,case_capacity=4,top_k=top_k,use_glocal_weightor=False)
    m.add_cases(torch.zeros(3,2),torch.tensor([0.,1.,2.]))
    return m


def test_absolute_budget_and_deterministic_Q_selection():
    m=critic()
    result=constrained_retain(m,TrainingReference(torch.zeros(1,2),'training_gae',targets=torch.tensor([1.])),keep_capacity=1)
    assert result['accepted']==[1]
    assert result['final_ids']==[0,2]
    assert not result['reached'] and result['actual_capacity']==2
    assert result['final_loss']==0 and result['original_loss']==0
    assert len(result['trials'])==5


def test_unknown_original_exposure_and_explicit_protection():
    m=critic(top_k=1)
    with torch.no_grad():m.nnknn_model.biases[:3].copy_(torch.tensor([0.,4.,1.]))
    result=constrained_retain(m,TrainingReference(torch.zeros(1,2),'training_gae',targets=torch.tensor([2.])),keep_capacity=1,loss_budget=1.)
    assert result['accepted']==[1] and result['final_ids']==[0,2]
    assert result['exposure']==[0,1,0] and not result['reached']
    m=critic()
    result=constrained_retain(m,TrainingReference(torch.zeros(1,2),'training_gae',targets=torch.tensor([1.])),keep_capacity=2,protected_ids=[1])
    assert not result['accepted'] and result['actual_capacity']==3


def test_actor_coverage_and_warmup_and_MC_guards():
    m=NNKNNPolicyNetwork(2,2,case_capacity=4,top_k=4,min_cases_per_action=2,use_glocal_weightor=False)
    m.add_cases(torch.zeros(4,2),torch.tensor([0,0,1,1]))
    ref=TrainingReference(torch.zeros(1,2),'actor_policy_surrogate',actions=torch.tensor([0]),
        advantages=torch.tensor([1.]),behavior_epsilons=torch.tensor([.1]),policy_ready=torch.tensor([True]))
    result=constrained_retain(m,ref,keep_capacity=3,loss_budget=10.)
    assert not result['accepted'] and m.action_counts().tolist()==[2,2]
    ref.policy_ready=torch.tensor([False])
    with pytest.raises(ValueError,match='warmup'):constrained_retain(m,ref,keep_capacity=3)
    c=critic()
    with pytest.raises(ValueError,match='MC is diagnostic'):
        constrained_retain(c,TrainingReference(torch.zeros(1,2),'independent_mc',targets=torch.tensor([1.])),keep_capacity=2)


def test_compaction_identity_Adam_and_lagged_target_alignment():
    m=critic();target=_build_nnknn_target_value_model(m,device=torch.device('cpu'))
    optimizer=torch.optim.Adam(m.parameters(),lr=.01)
    for p in m.parameters():p.grad=torch.ones_like(p)
    optimizer.step()
    old_ids=m.active_case_ids().clone()
    state=copy.deepcopy(optimizer.state[m.nnknn_model.biases])
    with torch.no_grad():
        m.nnknn_model.biases.zero_()
        target.nnknn_model.labels[:3].copy_(torch.tensor([[10.],[11.],[12.]]))
    constrained_retain(m,TrainingReference(torch.zeros(1,2),'training_gae',targets=torch.tensor([1.])),keep_capacity=2)
    _realign_optimizer_case_state(optimizer,m,old_ids)
    _align_nnknn_target_case_store(m,target)
    assert m.active_case_ids().tolist()==target.active_case_ids().tolist()==[0,2]
    assert target.nnknn_model.labels[:2].flatten().tolist()==[10.,12.]
    assert torch.equal(optimizer.state[m.nnknn_model.biases]['exp_avg'][:2],state['exp_avg'][[0,2]])
