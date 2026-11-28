import copy
import numpy as np
import pytest
import torch

from model.common import tracking
from model.t1 import records
from model.t1.pairwise_adaptation import (
    PairNetwork, pair_training_data, local_ridge_predict, classification_from_affinity,
    predict_pairs, forest_pair_predict,
)
from model.t1.pairwise_training import fit_neural, fit_forest


def fixture(kind, n=24):
    rng = np.random.default_rng(17)
    z = rng.normal(size=(n,3)).astype(np.float32)
    y = (z[:,0]>0).astype(int) if kind=="classification" else (2+z[:,0]-.3*z[:,1]+.6*z[:,2]).astype(np.float32)
    targets = np.eye(2,dtype=np.float32)[y] if kind=="classification" else y[:,None]
    context = dict(task_type=kind,case_z=z,case_targets=targets,prior=targets.mean(0) if kind=="classification" else None)
    for name,query in [("train",z),("val",rng.normal(size=(7,3)).astype(np.float32)),("test",rng.normal(size=(8,3)).astype(np.float32))]:
        distance = np.square(query[:,None]-z[None]).sum(-1)
        if name=="train":
            np.fill_diagonal(distance,np.inf)
        index = np.argsort(distance,axis=1)[:,:6]
        weights = np.exp(-np.take_along_axis(distance,index,axis=1)/2.)
        weights /= weights.sum(1,keepdims=True)
        context[name] = dict(query_z=query,indices=index,weights=weights.astype(np.float32),
            p0=(weights[...,None]*targets[index]).sum(1).astype(np.float32),
            target=(query[:,0]>0).astype(int) if kind=="classification" else (2+query[:,0]-.3*query[:,1]+.6*query[:,2]).astype(np.float32))
    return context


@pytest.mark.parametrize("kind",["classification","regression"])
def test_pairs_have_equal_query_mass_no_self_and_train_labels_only(kind):
    context = fixture(kind)
    pairs = pair_training_data(context,seed=5)
    assert np.all(pairs["query_index"] != pairs["case_index"])
    mass = np.bincount(pairs["query_index"],weights=pairs["weight"])
    assert np.allclose(mass,mass[0])
    changed = copy.deepcopy(context)
    changed["val"]["target"][:] = 99
    changed["test"]["target"][:] = -99
    same = pair_training_data(changed,seed=5)
    assert all(np.array_equal(v,same[k]) for k,v in pairs.items())


@pytest.mark.parametrize("kind",["classification","regression"])
def test_pair_network_eval_swap_properties(kind):
    model = PairNetwork(3,kind,dropout=.1).eval()
    a,b = torch.randn(9,3),torch.randn(9,3)
    if kind=="classification":
        assert torch.equal(model(a,b),model(b,a))
        assert ((model(a,b)>0)&(model(a,b)<1)).all()
    else:
        assert torch.equal(model(a,b),-model(b,a))
        assert torch.count_nonzero(model(a,a))==0
    assert torch.equal(model(a,b),model(a,b))


def test_local_ridge_recovers_affine_function_and_constant_targets():
    context = fixture("regression")
    prediction = local_ridge_predict(context,context["val"],1e-9)
    np.testing.assert_allclose(prediction[:,0],context["val"]["target"],atol=2e-5)
    context["case_targets"][:] = 2.
    context["val"]["p0"][:] = 2.
    np.testing.assert_allclose(local_ridge_predict(context,context["val"],.1),np.full((7,1),2.),atol=1e-12,rtol=0)


def test_local_ridge_matches_direct_primal_solution():
    context = fixture("regression")
    part = context["val"]
    expected = []
    for q,i,w,p0 in zip(part["query_z"],part["indices"],part["weights"],part["p0"]):
        x,y = context["case_z"][i].astype(float),context["case_targets"][i].astype(float)
        w = w.astype(float)
        xm,ym = (w[:,None]*x).sum(0),(w[:,None]*y).sum(0)
        xc,yc = x-xm,y-ym
        slope = np.linalg.solve(xc.T@(w[:,None]*xc)+.1*np.eye(x.shape[1]),xc.T@(w[:,None]*yc))
        expected.append(p0+(q-xm)@slope)
    np.testing.assert_allclose(local_ridge_predict(context,part,.1),expected,atol=1e-10)


def test_classification_posteriors_are_valid_and_identity_affinity_is_retrieval():
    targets = np.eye(3)[np.array([[0,1],[2,0]])]
    weights = np.array([[.2,.8],[.7,.3]])
    prior = np.array([.2,.5,.3])
    result = classification_from_affinity(np.ones((2,2)),targets,weights,prior)
    np.testing.assert_allclose(result,(weights[...,None]*targets).sum(1))
    result = classification_from_affinity(np.array([[.4,.1],[0.,.9]]),targets,weights,prior)
    assert np.all(result>=0)
    np.testing.assert_allclose(result.sum(1),1.)
    context = fixture("classification")
    prediction = local_ridge_predict(context,context["val"],.01)
    assert np.all(prediction>=0)
    np.testing.assert_allclose(prediction.sum(1),1.)


@pytest.mark.parametrize("kind",["classification","regression"])
def test_prediction_does_not_read_query_target(kind):
    context = fixture(kind)
    model = PairNetwork(3,kind).eval()
    before = predict_pairs(model,context,context["test"],family="pair_neural")
    del context["test"]["target"]
    after = predict_pairs(model,context,context["test"],family="pair_neural")
    np.testing.assert_array_equal(before,after)


def test_individual_differences_reconstruct_query_target_before_aggregation():
    context = fixture("regression")
    class KnownDifference(torch.nn.Module):
        def forward(self,a,b):
            return ((a-b)*a.new_tensor([1.,-.3,.6])).sum(1,keepdim=True)
    part = context["test"]
    prediction = predict_pairs(KnownDifference(),context,part,family="pair_neural")
    np.testing.assert_allclose(prediction[:,0],part["target"],atol=1e-6)
    permuted = {**part,"indices":part["indices"][:,::-1].copy(),"weights":part["weights"][:,::-1].copy()}
    np.testing.assert_allclose(prediction,predict_pairs(KnownDifference(),context,permuted,family="pair_neural"),atol=1e-12)


@pytest.mark.parametrize("kind",["classification","regression"])
@pytest.mark.parametrize("family",["pair_neural","pair_trees"])
def test_exact_interrupted_recovery(tmp_path,monkeypatch,kind,family):
    monkeypatch.setenv("NNKNN_RUNS",str(tmp_path))
    context = fixture(kind)
    pairs = pair_training_data(context,seed=7,random_anchors=2)
    config = (dict(hidden=[8,4],dropout=.1,lr=.001,weight_decay=1e-4,grad_clip=5.,batch_size=32,epochs=4,patience=20)
        if family=="pair_neural" else dict(n_estimators=8,tree_step=4,max_depth=4,max_features=.7,min_samples_leaf=2))
    fn = fit_neural if family=="pair_neural" else fit_forest
    with tracking.open_run(records.key("pr","full"),config=dict(kind=kind,family=family)) as run:
        expected,info = fn(context,pairs,config,seed=3,run=run)
    key = records.key("pr","resume")
    with pytest.raises(RuntimeError,match="interrupt"):
        with tracking.open_run(key,config=dict(kind=kind,family=family)) as run:
            save = run.save_resume
            def stop(*args,**kwargs):
                save(*args,**kwargs)
                if kwargs["step"]==(2 if family=="pair_neural" else 4):
                    raise RuntimeError("interrupt")
            monkeypatch.setattr(run,"save_resume",stop)
            fn(context,pairs,config,seed=3,run=run)
    with tracking.open_run(key,config=dict(kind=kind,family=family)) as run:
        actual,resumed = fn(context,pairs,config,seed=3,run=run)
    assert info==resumed
    np.testing.assert_array_equal(
        predict_pairs(expected,context,context["test"],family=family),
        predict_pairs(actual,context,context["test"],family=family))
    if family=="pair_trees":
        a,b = pairs["a"][:8],pairs["b"][:8]
        ab,ba = forest_pair_predict(actual,a,b,kind),forest_pair_predict(actual,b,a,kind)
        np.testing.assert_array_equal(ab,ba if kind=="classification" else -ba)


def test_selection_and_complete_unit_reuse(tmp_path,monkeypatch):
    from tools import t1_pairwise_redesign as driver
    monkeypatch.setenv("NNKNN_RUNS",str(tmp_path))
    context = fixture("regression")
    context["core_hash"] = "frozen"
    key = driver.key_for("pr","regression__toy",1,"ctx")
    with tracking.open_run(key,config={"test":"context"}) as run:
        run.save_state("context",records.storable(context))
    unit = tracking.load_unit(key,config={"test":"context"},arrays=False)
    candidate = driver.run_candidate("pr","regression__toy",1,"local_ridge","r0",unit,context)
    driver.evaluate_selected("pr","regression__toy",1,"local_ridge",[candidate],context)
    def forbid(*a,**k):
        raise AssertionError("complete unit recomputed")
    monkeypatch.setattr(driver,"predict_model",forbid)
    driver.run_candidate("pr","regression__toy",1,"local_ridge","r0",unit,context)
    driver.evaluate_selected("pr","regression__toy",1,"local_ridge",[candidate],context)
    rows=[dict(candidate="a",val_error=.2,val_task_loss=.1,test_error=0),
          dict(candidate="b",val_error=.1,val_task_loss=.2,test_error=10)]
    assert driver.choose_candidate(rows)["candidate"]=="b"
    rows[0]["test_error"],rows[1]["test_error"]=100,-100
    assert driver.choose_candidate(rows)["candidate"]=="b"


def test_policy_ties_keep_retrieval_and_do_not_consult_test():
    from tools.t1_pairwise_redesign_summary import choose_policy, choose_nncdh
    rows = [dict(method="pair_neural", val_error=.1, test_error=0),
        dict(method="retrieval", val_error=.1, test_error=1)]
    assert choose_policy(rows)["method"] == "retrieval"
    rows[0]["val_error"] = .09
    assert choose_policy(rows)["method"] == "pair_neural"
    old = [dict(adapter="original", scheme="sequential", dropout=0., val_task_loss=.2, val_error=.1, test_error=0),
        dict(adapter="query_aware", scheme="joint", dropout=.1, val_task_loss=.1, val_error=.2, test_error=1)]
    assert choose_nncdh(old)["adapter"] == "query_aware"
    old[0]["test_error"], old[1]["test_error"] = 100, -100
    assert choose_nncdh(old)["adapter"] == "query_aware"


def test_behavior_totals_keep_harm_severity_and_weight_pooled_mse():
    from tools.t1_pairwise_redesign_summary import add_behavior, behavior_totals
    behavior, harms = [], []
    row = dict(task="regression__toy", task_type="regression", seed=1, method="pair_trees", unit="toy")
    add_behavior(row, dict(p0=np.array([0., 0.]), final=np.array([-1., 3.]), target=np.ones(2)), behavior, harms)
    add_behavior(dict(row, seed=2), dict(p0=np.zeros(1), final=np.ones(1), target=np.ones(1)), behavior, harms)
    total, = behavior_totals(behavior)
    assert total["n"] == 3
    assert total["count_wrong_direction"] == 1
    assert total["count_overshoot_harms"] == 1
    assert total["total_harm_sse"] == 6.
    assert total["total_benefit_sse"] == 1.
    assert total["pooled_post_error"] == pytest.approx(np.sqrt(8/3))
    assert len(harms) == 2
