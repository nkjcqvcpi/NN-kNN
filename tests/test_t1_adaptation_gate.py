import copy
import numpy as np
import pytest
import torch
from torch import nn

from model.t1.adaptation_gate import ApplyOrSkipAdapter, gate_loss, objective, train_continuation
from tools.t1_adaptation_gate import static_choice


def inputs():
    return dict(x=torch.tensor([[1.,0.],[0.,1.],[-1.,1.],[1.,-1.]]),
                p0=torch.zeros(4,1),statistics=torch.tensor([[.1,.2,.5,.3]]).repeat(4,1),
                target=torch.tensor([1.,-.5,.2,-1.]))


def network(n=1):
    return nn.Sequential(nn.Linear(2,4),nn.Tanh(),nn.Linear(4,n))


def settings():
    return dict(score_scale=1.,v_post=1.,v_delta=1.,v_small=0.,free_radius=None,
                lr=.001,weight_decay=0.,epochs=3,patience=2,batch_size=2)


def test_gate_is_exact_full_or_zero_and_keeps_proposal_unchanged():
    torch.manual_seed(9)
    net=network()
    data=inputs()
    model=ApplyOrSkipAdapter(net,"regression","probability_gate")
    model.fit_normalizer(data)
    with torch.no_grad(): model.gate.bias.fill_(1)
    on=model(data["x"],data["p0"],data["statistics"])
    assert torch.equal(on.proposal,net(data["x"]))
    assert torch.equal(on.final,on.full)
    with torch.no_grad(): model.gate.bias.fill_(-1)
    off=model(data["x"],data["p0"],data["statistics"])
    assert torch.equal(off.final,data["p0"])
    assert torch.equal(off.proposal,on.proposal)


def test_classification_keeps_original_tanh_proposal():
    data=inputs()
    data["p0"]=torch.tensor([[.4,.6]]).repeat(4,1)
    net=network(2)
    model=ApplyOrSkipAdapter(net,"classification","full_control")
    model.fit_normalizer(data)
    p=model(data["x"],data["p0"],data["statistics"])
    assert torch.equal(p.proposal,net(data["x"]).tanh())


def test_cost_weighting_can_reverse_a_majority_of_small_wins():
    # Two tiny gains, one expensive mistake: majority-help probability and
    # expected loss correctly recommend opposite constant decisions.
    before=torch.tensor([.2,.2,.2],requires_grad=True)
    full=torch.tensor([.1,.1,1.2],requires_grad=True)
    ordinary=torch.zeros(3,requires_grad=True)
    weighted=torch.zeros(3,requires_grad=True)
    gate_loss(ordinary,before,full,weighted=False).backward()
    gate_loss(weighted,before,full,weighted=True).backward()
    assert ordinary.grad.sum()<0
    assert weighted.grad.sum()>0
    assert before.grad is None and full.grad is None


def test_gate_and_residual_heads_both_learn_when_gate_is_closed():
    torch.manual_seed(4)
    data=inputs()
    model=ApplyOrSkipAdapter(network(),"regression","utility_gate")
    model.fit_normalizer(data)
    with torch.no_grad(): model.gate.bias.fill_(-1)
    loss,_=objective(model,data,settings())
    loss.backward()
    assert model.gate.weight.grad.abs().sum()>0
    assert model.network[-1].weight.grad.abs().sum()>0
    assert model.network[0].weight.grad.abs().sum()>0


def test_normalization_does_not_read_query_targets_or_update_on_validation():
    data=inputs()
    model=ApplyOrSkipAdapter(network(),"regression","probability_gate")
    model.fit_normalizer(data)
    before=copy.deepcopy(model.state_dict())
    data["target"]=torch.ones_like(data["target"])*1000
    model(data["x"]*10,data["p0"],data["statistics"])
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())


def test_static_selection_uses_validation_and_prefers_less_adaptation_on_ties():
    data=inputs()
    data["p0"]=torch.tensor([[.9,.1]]).repeat(4,1)
    data["target"]=torch.zeros(4,dtype=torch.long)
    full=torch.tensor([[.8,.2]]).repeat(4,1)
    choices=static_choice("classification",data,full,settings(),data["statistics"])
    assert choices["global_switch"]=={"rule":"scale","parameter":0.}
    assert choices["global_shrink"]=={"rule":"scale","parameter":0.}


def test_training_requires_recording():
    with pytest.raises(TypeError,match="recorded run"):
        train_continuation(ApplyOrSkipAdapter(network(),"regression","full_control"),
                           inputs(),inputs(),settings(),1,run=None)


def test_recorded_training_saves_selected_weights_and_retains_resume(tmp_path,monkeypatch):
    from model.common.tracking import open_run,load_unit
    from model.t1 import records
    monkeypatch.setenv("NNKNN_RUNS",str(tmp_path))
    key=records.key("gtest","unit")
    torch.manual_seed(1)
    with open_run(key,config={"test":"gate"}) as run:
        model=ApplyOrSkipAdapter(network(),"regression","utility_gate")
        info=train_continuation(model,inputs(),inputs(),settings(),2,run=run)
        records.save_rng(run)
    saved=load_unit(key,config={"test":"gate"},arrays=False)
    assert saved is not None
    checkpoint=saved.load_state("checkpoint")
    assert checkpoint["selected_epoch"]==info["selected_epoch"]
    assert all(torch.equal(v,checkpoint["model"][k]) for k,v in model.state_dict().items())
    assert (saved.info.path/"weights"/"resume"/"last.pt").exists()
