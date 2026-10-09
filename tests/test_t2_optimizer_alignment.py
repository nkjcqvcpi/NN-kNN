import copy

import pytest
import torch

from model.nnknn_rl_workflow import (
    NNKNNValueNetwork, _realign_optimizer_case_state, make_nnknn_rl_config,
    _build_actor_critic_models, _build_joint_nnknn_rl_optimizer,
)


def test_actual_adam_moments_follow_surviving_ids_and_zero_new_rows():
    m=NNKNNValueNetwork(2,case_capacity=4,trainable_value_labels=True)
    m.add_cases(torch.tensor([[0.,0.],[1.,1.],[2.,2.]]),torch.tensor([1.,2.,3.]))
    optimizer=torch.optim.Adam(m.parameters(),lr=.01,amsgrad=True)
    for p in m.parameters():
        p.grad=torch.arange(p.numel(),dtype=p.dtype).reshape_as(p)+1
    optimizer.step()
    before_ids=m.active_case_ids().clone()
    names={'nnknn_model.biases','nnknn_model.labels','nnknn_model.glocal_weights'}
    saved={name:copy.deepcopy(optimizer.state[p]) for name,p in m.named_parameters() if name in names}
    m._compact_cases([1,2])
    m.add_cases(torch.tensor([[3.,3.]]),torch.tensor([4.]))
    event=_realign_optimizer_case_state(optimizer,m,before_ids)
    assert event['preserved_case_ids']==[1,2]
    assert event['new_case_ids']==[3] and event['removed_case_ids']==[0]
    assert event['tensor_states']>=9
    for name,p in m.named_parameters():
        if name not in names:continue
        state=optimizer.state[p]
        assert torch.equal(state['step'],saved[name]['step'])
        for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            assert torch.equal(state[key][:2],saved[name][key][[1,2]])
            assert torch.count_nonzero(state[key][2:])==0
    # A real continuation step uses retained moments instead of discarding them.
    for p in m.parameters():p.grad=torch.ones_like(p)
    optimizer.step()
    assert all(float(optimizer.state[p]['step'])==2 for p in m.parameters())


def test_joint_optimizer_alignment_keeps_other_role_and_shared_trunk_untouched():
    cfg=make_nnknn_rl_config('smoke',case_capacity=4,critic_type='nnknn')
    actor,critic=_build_actor_critic_models(2,2,cfg,torch.device('cpu'))
    actor.add_cases(torch.tensor([[0.,0.],[1.,1.]]),torch.tensor([0,1]))
    critic.add_cases(torch.tensor([[0.,0.],[1.,1.]]),torch.tensor([1.,2.]))
    optimizer=_build_joint_nnknn_rl_optimizer(actor,critic,base_lr=.01,case_lr=.02)
    for group in optimizer.param_groups:
        for p in group['params']:p.grad=torch.ones_like(p)
    optimizer.step()
    actor_states={p:copy.deepcopy(optimizer.state[p]) for name,p in actor.named_parameters()}
    ids=critic.active_case_ids().clone()
    critic._compact_cases([1])
    _realign_optimizer_case_state(optimizer,critic,ids)
    for p,state in actor_states.items():
        assert optimizer.state[p].keys()==state.keys()
        assert all(torch.equal(optimizer.state[p][key],value) for key,value in state.items())
    with pytest.raises(ValueError,match='unique'):
        _realign_optimizer_case_state(optimizer,critic,torch.tensor([1,1]))
