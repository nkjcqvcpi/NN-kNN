import copy
from dataclasses import replace
import pytest
import torch
from model.common import tracking
from model.common.core import CoreConfig, build_model, make_optimizer
from model.t1 import records
from model.t1.data import T1Data
from model.t1.sync import SyncConfig, train_synchronized


def fixture(task="regression", design="residual", objective="summed"):
    x=torch.tensor([[-1.,.2],[-.7,-.1],[-.2,.3],[.3,-.2],[.7,.1],[1.2,.4]])
    y=(x[:,0]>0).long() if task=="classification" else x[:,0]+.3*x[:,1]
    data=T1Data("tiny",task,x,y,x+.01,y,x+.02,y,num_classes=2 if task=="classification" else None)
    core=CoreConfig(task_type=task,representation="mlp",mlp_dims=(8,),batch_size=3,top_k=3,bias_init="manual",seed=7)
    cfg=SyncConfig("joint_rr",1.,.7,.1,1.,1.,0.,epochs=4,patience=4,adapter_warmup_epochs=1,
        hidden_dims=(8,4),seed=7,diagnostics_every=0,adapter_design=design,joint_objective=objective,
        output_mode="nominal_residual_scores" if task=="classification" else "aggregate_regression")
    return build_model(x,y,core,data.num_classes),core,data,cfg


@pytest.mark.parametrize("task",["classification","regression"])
@pytest.mark.parametrize("design",["residual","query_aware","query_aware_no_prediction"])
@pytest.mark.parametrize("scheme",["sequential","alternating"])
def test_shared_core_and_component_pass_budgets(task,design,scheme):
    model,core,data,cfg=fixture(task,design)
    initial=copy.deepcopy(model.state_dict())
    flags=[p.requires_grad for p in model.parameters()]
    _,info=train_synchronized(model,data,core,cfg,None,near_scale=1.,comparison_scheme=scheme)
    assert len(info["history"])==4
    assert sum(h["adapter_updates"] for h in info["history"])==8
    assert sum(h["retrieval_updates"] for h in info["history"])==(6 if scheme=="alternating" else 0)
    assert [p.requires_grad for p in model.parameters()]==flags
    if scheme=="sequential":
        assert all(torch.equal(value,model.state_dict()[key]) for key,value in initial.items())
    else:
        assert [h["phase"] for h in info["history"]]==["adapter","alternating","alternating","alternating"]


@pytest.mark.parametrize("scheme",["sequential","alternating"])
@pytest.mark.parametrize("objective",["summed","separated"])
def test_zero_retrieval_rate_matches_joint_adapter_updates(scheme,objective):
    # With retrieval held stationary, all schemes must give the same adapter
    # updates. This catches accidentally dropping w_post from summed training.
    model,core,data,cfg=fixture("regression","query_aware",objective)
    cfg=replace(cfg,epochs=3,patience=3)
    initial=copy.deepcopy(model.state_dict())
    def train(mode):
        model.load_state_dict(initial)
        model.eval(); model._invalidate_case_cache()
        opt=make_optimizer(model,core)
        for group in opt.param_groups: group["lr"]=0.
        return train_synchronized(model,data,core,cfg,None,near_scale=1.,core_optimizer=opt,comparison_scheme=mode)
    expected,info=train(None)
    actual,other=train(scheme)
    assert other["best_epoch"]==info["best_epoch"]
    for key,value in expected.state_dict().items():
        torch.testing.assert_close(value,actual.state_dict()[key],rtol=2e-6,atol=2e-7)
    assert [h["val_L_post"] for h in info["history"]]==pytest.approx([h["val_L_post"] for h in other["history"]],rel=2e-6)


@pytest.mark.parametrize("scheme",["sequential","alternating"])
def test_recorded_cycle_resume_matches_uninterrupted(tmp_path,monkeypatch,scheme):
    monkeypatch.setenv("NNKNN_RUNS",str(tmp_path))
    model,core,data,cfg=fixture("classification","query_aware_no_prediction")
    initial=copy.deepcopy(model.state_dict())
    def train(run):
        model.load_state_dict(initial); model._invalidate_case_cache()
        return train_synchronized(model,data,core,cfg,None,near_scale=1.,comparison_scheme=scheme,run=run)
    with tracking.open_run(records.key("sc","full"),config={"scheme":scheme}) as run:
        expected,info=train(run)
    final_core=copy.deepcopy(model.state_dict())
    key=records.key("sc","resume")
    with pytest.raises(RuntimeError,match="interrupt"):
        with tracking.open_run(key,config={"scheme":scheme}) as run:
            save=run.save_resume
            def stop(*args,**kwargs):
                save(*args,**kwargs)
                if kwargs["step"]==2: raise RuntimeError("interrupt")
            monkeypatch.setattr(run,"save_resume",stop)
            train(run)
    with tracking.open_run(key,config={"scheme":scheme}) as run:
        actual,resumed=train(run)
    assert info["history"]==resumed["history"]
    assert info["best_epoch"]==resumed["best_epoch"]
    assert all(torch.equal(v,actual.state_dict()[k]) for k,v in expected.state_dict().items())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in final_core.items())


def test_comparison_override_rejects_unmatched_legacy_schedule():
    model,core,data,cfg=fixture()
    with pytest.raises(ValueError,match="controlled scheme"):
        train_synchronized(model,data,core,replace(cfg,schedule="alternating_rr",adapter_warmup_epochs=0),None,
                           near_scale=1.,comparison_scheme="alternating")


def test_scheme_selection_uses_validation_only_and_deterministic_ties():
    from tools.t1_adapter_training_schemes import choose_scheme
    rows=[dict(scheme=s,val_task_loss=l,val_error=e,test_error=t) for s,l,e,t in
          (("sequential",.2,.3,0.),("alternating",.1,.2,1.),("joint",.1,.1,.9))]
    assert choose_scheme(rows)["scheme"]=="joint"
    rows[0]["test_error"],rows[2]["test_error"]=100.,0.
    assert choose_scheme(rows)["scheme"]=="joint"
    for row in rows: row.update(val_task_loss=.1,val_error=.1)
    assert choose_scheme(list(reversed(rows)))["scheme"]=="sequential"


def test_component_counts_distinguish_passes_from_parameter_updates():
    from tools.t1_adapter_training_schemes import component_counts
    history=[dict(epoch=1,phase="adapter",optimizer_updates=3),
             dict(epoch=2,phase="joint",optimizer_updates=3)]
    result=component_counts(history,1)
    assert result["retrieval_updates_selected"]==0
    assert result["adapter_updates_selected"]==3
    assert result["retrieval_updates_executed"]==3
    assert result["adapter_updates_executed"]==6
    history[1]=dict(epoch=2,phase="alternating",optimizer_updates=6,retrieval_updates=3,adapter_updates=3)
    assert component_counts(history,1)==result
