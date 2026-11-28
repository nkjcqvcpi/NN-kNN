import copy
from dataclasses import replace
import pytest
import torch

from model.common import tracking
from model.common.core import CoreConfig, build_model
from model.nn_cdh import ClassificationNNCDHAdapter, NNCDHAdapter
from model.t1 import records
from model.t1.data import T1Data
from model.t1.query_aware import (QueryAwareClassificationAdapter, QueryAwareRegressionAdapter,
    QUERY_AWARE_DESIGNS, query_adapter_type)
from model.t1.reuse import ReuseConfig, train_classification_adapter, train_regression_adapter, evaluate_reuse
from model.t1.sync import SyncConfig, train_synchronized
from model.t1.outcomes import final_prediction


def fixture(task,representation="raw"):
    x = torch.tensor([[-1.,.2],[-.7,-.1],[-.2,.3],[.3,-.2],[.7,.1],[1.2,.4]])
    y = (x[:,0]>0).long() if task=="classification" else x[:,0]+x[:,1]*.3
    data = T1Data("tiny", task, x, y, x+.01, y, x+.02, y,
                  num_classes=2 if task=="classification" else None)
    cc = CoreConfig(task_type=task, batch_size=3, top_k=3, bias_init="manual", seed=7,
                    representation=representation,mlp_dims=(8,))
    return build_model(x, y, cc, data.num_classes), cc, data


def sync_config(task, schedule, design="query_aware"):
    return SyncConfig(schedule,1.,.5,.1,1.,1.,0.,epochs=4,patience=4,block_epochs=1,
                      hidden_dims=(8,4),adapter_design=design, diagnostics_every=0,
                      output_mode="nominal_residual_scores" if task=="classification" else "aggregate_regression",
                      seed=7)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_zero_query_connections_preserve_original_function_and_rng(task):
    old_type = ClassificationNNCDHAdapter if task=="classification" else NNCDHAdapter
    new_type = QueryAwareClassificationAdapter if task=="classification" else QueryAwareRegressionAdapter
    args = (2,2) if task=="classification" else (2,1)
    torch.manual_seed(21)
    old = old_type(*args,hidden_dims=(8,4))
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(21)
    new = new_type(*args,hidden_dims=(8,4))
    assert torch.equal(old_rng,torch.get_rng_state())
    q = torch.randn(7,2)
    cases = torch.randn(4,2)
    weights = torch.randn(7,4).softmax(1)
    p0 = torch.randn(7,args[1])
    if task=="classification":
        expected = old(q-weights@cases,p0)[1]
        actual = new(q-weights@cases,p0,query_features=q)[1]
        first = new.net[0]
    else:
        labels = torch.randn(4,1)
        expected = old.forward_aggregate(q,cases,labels,weights)
        actual = new.forward_aggregate(q,cases,labels,weights)
        first = new.adapt_net_final.model[0]
    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)
    actual.square().sum().backward()
    assert first.weight.grad[:,-2:].abs().sum()>0


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("schedule", ["independent", "alternating_rr", "joint_rr"])
@pytest.mark.parametrize("design", sorted(QUERY_AWARE_DESIGNS))
def test_training_schedules_update_query_weights_and_route_actual_output(task,schedule,design):
    model, cc, data = fixture(task)
    adapter, info = train_synchronized(model,data,cc,sync_config(task,schedule,design),None,near_scale=1.)
    first = adapter.net[0] if task=="classification" else adapter.adapt_net_final.model[0]
    assert first.weight[:,-2:].abs().sum()>0
    assert info["best_epoch"]>0
    model.eval()
    r = model.retrieve(data.X_val,exclude_identical=False)
    diagnostics = {}
    pre, final, _ = final_prediction(model,data.X_val,r,adapter=adapter,diagnostics=diagnostics)
    torch.testing.assert_close(pre+diagnostics["residual"],final)
    torch.testing.assert_close(diagnostics["query_features"],r["query_features"])


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("design", sorted(QUERY_AWARE_DESIGNS))
def test_frozen_reuse_keeps_core_unchanged(task,design):
    model, cc, data = fixture(task)
    before = copy.deepcopy(model.state_dict())
    cfg = ReuseConfig("nominal_residual_scores","combined",1.,1.,"softmax",
                      epochs=4,patience=4,batch_size=3,hidden_dims=(8,4),seed=7,adapter_design=design)
    train = train_classification_adapter if task=="classification" else train_regression_adapter
    adapter, info = train(model,data,cfg)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    r = model.retrieve(data.X_val,exclude_identical=False)
    pre, final, _ = final_prediction(model,data.X_val,r,adapter=adapter)
    assert final.shape == pre.shape
    if task=="classification":
        result = evaluate_reuse(model,adapter,data.X_val,data.y_val,cfg)
        assert result["accuracy_post"] == pytest.approx(float((final.argmax(1)==data.y_val).float().mean()))


def test_explicit_query_required_and_unsupported_modes_rejected():
    adapter = QueryAwareClassificationAdapter(2,2)
    with pytest.raises(ValueError,match="query representation"):
        adapter(torch.zeros(3,2),torch.ones(3,2)/2)
    with pytest.raises(ValueError,match="additive"):
        QueryAwareClassificationAdapter(2,2,output_mode="logit_residual")


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("design", sorted(QUERY_AWARE_DESIGNS))
def test_retrieval_events_record_query_and_actual_residual(task,design):
    from model.t1.core import retrieval_events
    model,cc,data=fixture(task)
    adapter=query_adapter_type(task,design)(2,2 if task=="classification" else 1)
    model.eval(); adapter.eval()
    with torch.no_grad():
        r=model.retrieve(data.X_test,exclude_identical=False)
        pre,final,_=final_prediction(model,data.X_test,r,adapter=adapter)
        events=retrieval_events(model,data.X_test,data.y_test,stream="test",run_id="test",adapter=adapter)
    assert len(events)==len(data.y_test)
    torch.testing.assert_close(torch.tensor([e["predicted_residual"] for e in events]),final-pre)
    torch.testing.assert_close(torch.tensor([e["query_features"] for e in events]),r["query_features"])
    assert all(e["adapter_design"]==design for e in events)
    if task=="classification":
        torch.testing.assert_close(torch.tensor([e["final_values"] for e in events]),final)
    else:
        torch.testing.assert_close(torch.tensor([e["final_prediction"] for e in events]),final[:,0])


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("design", sorted(QUERY_AWARE_DESIGNS))
def test_mlp_representation_warmup_does_not_reuse_a_backward_graph(task,design):
    model,cc,data=fixture(task,"mlp")
    cfg=replace(sync_config(task,"joint_rr",design),adapter_warmup_epochs=2)
    adapter,info=train_synchronized(model,data,cc,cfg,None,near_scale=1.)
    assert len(info["history"])==4
    assert info["history"][0]["phase"]=="adapter"


@pytest.mark.parametrize("kind", ["sync", "reuse"])
@pytest.mark.parametrize("design", sorted(QUERY_AWARE_DESIGNS))
def test_interruption_resumes_exactly(tmp_path,monkeypatch,kind,design):
    monkeypatch.setenv("NNKNN_RUNS",str(tmp_path))
    model, cc, data = fixture("regression")
    initial = copy.deepcopy(model.state_dict())
    def train(run):
        model.load_state_dict(initial)
        model._invalidate_case_cache()
        if kind=="sync":
            return train_synchronized(model,data,cc,sync_config("regression","joint_rr",design),
                                      None,near_scale=1.,run=run)
        cfg = ReuseConfig("nominal_residual_scores","combined",1.,1.,"softmax",
                          epochs=4,patience=4,batch_size=3,hidden_dims=(8,4),seed=7,adapter_design=design)
        return train_regression_adapter(model,data,cfg,run=run)
    with tracking.open_run(records.key("qa","full"),config={"kind":kind}) as run:
        expected, info = train(run)
    expected_core = copy.deepcopy(model.state_dict())
    key = records.key("qa","resume")
    with pytest.raises(RuntimeError,match="interrupt"):
        with tracking.open_run(key,config={"kind":kind}) as run:
            save = run.save_resume
            def stop(*args,**kwargs):
                save(*args,**kwargs)
                if kwargs["step"]==2:
                    raise RuntimeError("interrupt")
            monkeypatch.setattr(run,"save_resume",stop)
            train(run)
    with tracking.open_run(key,config={"kind":kind}) as run:
        actual, resumed = train(run)
    assert info["best_epoch"] == resumed["best_epoch"]
    assert info["history"] == resumed["history"]
    assert all(torch.equal(v,actual.state_dict()[k]) for k,v in expected.state_dict().items())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected_core.items())


@pytest.mark.parametrize("family", ["mlp", "tabm"])
def test_neural_baseline_resume_keeps_selected_weights(tmp_path,monkeypatch,family):
    from tools.t1_grid_search import fit_candidate
    monkeypatch.setenv("NNKNN_RUNS",str(tmp_path))
    _,_,data=fixture("regression")
    allowed=dict(X_train=data.X_train,y_train=data.y_train,X_val=data.X_val,y_val=data.y_val,
                 task=data.task_type,classes=None)
    params=dict(width=8,layers=2,dropout=.1,lr=.001,weight_decay=0.) if family=="mlp" else dict(d_block=8,n_blocks=2,k=2,lr=.001,weight_decay=0.)
    with tracking.open_run(records.key("qb","full"),config={"family":family}) as run:
        expected,info=fit_candidate(family,params,allowed,7,4,4,run=run)
    unrecorded,plain_info=fit_candidate(family,params,allowed,7,4,4)
    assert info==plain_info
    assert all(torch.equal(v,unrecorded.state_dict()[k]) for k,v in expected.state_dict().items())
    key=records.key("qb","resume")
    with pytest.raises(RuntimeError,match="interrupt"):
        with tracking.open_run(key,config={"family":family}) as run:
            save=run.save_resume
            def stop(*args,**kwargs):
                save(*args,**kwargs)
                if kwargs["step"]==2: raise RuntimeError("interrupt")
            monkeypatch.setattr(run,"save_resume",stop)
            fit_candidate(family,params,allowed,7,4,4,run=run)
    with tracking.open_run(key,config={"family":family}) as run:
        actual,resumed=fit_candidate(family,params,allowed,7,4,4,run=run)
    assert info==resumed
    assert all(torch.equal(v,actual.state_dict()[k]) for k,v in expected.state_dict().items())


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_no_prediction_removes_columns_and_direct_cancellation_path(task):
    full_type=query_adapter_type(task,"query_aware")
    new_type=query_adapter_type(task,"query_aware_no_prediction")
    out_dim=3 if task=="classification" else 1
    torch.manual_seed(32)
    full=full_type(2,out_dim,hidden_dims=(8,4))
    state=torch.get_rng_state().clone()
    torch.manual_seed(32)
    new=new_type(2,out_dim,hidden_dims=(8,4))
    assert torch.equal(state,torch.get_rng_state())
    net1=full.net if task=="classification" else full.adapt_net_final.model
    net2=new.net if task=="classification" else new.adapt_net_final.model
    assert net2[0].in_features==4
    assert sum(p.numel() for p in full.parameters())-sum(p.numel() for p in new.parameters())==out_dim*8
    torch.testing.assert_close(net2[0].weight,torch.cat((net1[0].weight[:,:2],net1[0].weight[:,2+out_dim:]),1),rtol=0,atol=0)
    for a,b in zip(list(net1.parameters())[1:],list(net2.parameters())[1:]):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    q,dz=torch.randn(7,2),torch.randn(7,2)
    p=torch.randn(7,out_dim,requires_grad=True)
    changed=torch.randn_like(p)
    if task=="classification":
        residual,final=new(dz,p,query_features=q)
        other_residual,other=new(dz,changed,query_features=q)
        expected=full(dz,torch.zeros_like(p),query_features=q)[0]
        assert torch.autograd.grad(residual.sum(),p,allow_unused=True)[0] is None
    else:
        final=new.forward_differences(dz,p,q)
        other=new.forward_differences(dz,changed,q)
        residual=new.adapt_net_final(new.build_final_features(q,q-dz,p))
        other_residual=new.adapt_net_final(new.build_final_features(q,q-dz,changed))
        expected=full.forward_differences(dz,torch.zeros_like(p),q)
        assert torch.autograd.grad(residual.sum(),p,allow_unused=True)[0] is None
    torch.testing.assert_close(residual,other_residual,rtol=0,atol=0)
    torch.testing.assert_close(residual,expected)
    torch.testing.assert_close(final-p,other-changed)
    torch.testing.assert_close(other-final,changed-p)
