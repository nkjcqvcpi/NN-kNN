"""Synthetic autograd, directional-derivative and local-descent audit of T1 joint losses."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["NNKNN_DEVICE"] = "cpu"

import torch
import torch.nn.functional as F
from model.common.artifacts import environment_info, git_state, source_fingerprint, write_json
from model.common.core import CoreConfig, build_model, make_optimizer
from model.nn_cdh import ClassificationNNCDHAdapter, NNCDHAdapter
from model.regression_workflow import make_linear_regression_dataset, prepare_regression_split_for_training
from model.t1.calibration import calibrate_free_radius
from model.t1.sync import SyncConfig, _forward_terms, joint_backward
from model.t1.reuse import adapter_optimizer


def gradients(loss, params):
    raw = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach().clone() for p,g in zip(params,raw)]


def norm(gs):
    return math.sqrt(sum(float(g.double().square().sum()) for g in gs))


def cosine(a,b):
    den = norm(a)*norm(b)
    return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))/den if den else None


def objectives(T,cfg):
    R = cfg.w_pre*T["L_pre"] + cfg.w_post*T["L_post"] + cfg.w_near*T["L_near"]
    A = cfg.v_post*T["L_post"] + cfg.v_delta*T["L_delta"] + cfg.v_small*T["L_small"]
    return R,A,R+A


def adapter_for(model, task):
    torch.manual_seed(101)
    if task == "classification":
        return ClassificationNNCDHAdapter(model.feature_dim,2,0,(12,8),"nominal_residual_scores")
    a = NNCDHAdapter(model.feature_dim,1,(12,8))
    a.adapt_net_pair.requires_grad_(False)
    return a


def fixture(task, representation, masked, mcb):
    if task == "regression":
        raw = make_linear_regression_dataset(n=40,d=5,noise_scale=.2,seed=13)
        d = prepare_regression_split_for_training(raw["X"][:30],raw["y"][:30],raw["X"][30:],raw["y"][30:])
        X,y = d["X_train"],d["y_train"]
    else:
        X = torch.randn(30,5,generator=torch.Generator().manual_seed(13))
        y = (X[:,0]+.4*X[:,1]>0).long()
    cc = CoreConfig(task_type=task,representation=representation,mlp_dims=(16,8),mcb_enabled=mcb,
                    pre_topk_mask=masked,top_k=5,bias_init_k=5,seed=17,weight_decay=0.)
    m = build_model(X,y,cc,2 if task=="classification" else None)
    m.train()
    a = adapter_for(m,task)
    cfg = SyncConfig("joint_rr",1.,.5,0.,1.,1.,.1,joint_objective="summed",
                     output_mode="nominal_residual_scores" if task=="classification" else "aggregate_regression",
                     score_scale=10. if task=="classification" else 1.)
    radius,_ = calibrate_free_radius(m,s_task=.5,snapshot_step=0)
    ids = torch.arange(10)
    def terms():
        return _forward_terms(m,a,X[:10],y[:10],radius,output_mode=cfg.output_mode,near_scale=1.,
                              query_case_ids=ids,score_scale=cfg.score_scale)
    def support():
        return m.retrieve(X[:10],exclude_identical=False,query_case_ids=ids)["weights"].detach()>0
    return m,a,cc,cfg,terms,support,X[:10],ids,y[:10]


def perturbed(params, saved, direction, amount):
    with torch.no_grad():
        for p,s,g in zip(params,saved,direction):
            p.copy_(s+amount*g)


def probe(task, representation, masked, mcb):
    m,a,cc,cfg,terms,support,X,ids,y = fixture(task,representation,masked,mcb)
    named_r = [(n,p) for n,p in m.named_parameters() if p.requires_grad]
    named_a = [(n,p) for n,p in a.named_parameters() if p.requires_grad]
    rp,ap = [p for _,p in named_r],[p for _,p in named_a]
    T = terms()
    term_grads = {key:(gradients(T[key],rp),gradients(T[key],ap)) for key in ["L_pre","L_post","L_delta","L_small"]}
    assert norm(term_grads["L_pre"][1])==0
    assert norm(term_grads["L_post"][0])>1e-10 and norm(term_grads["L_post"][1])>1e-10
    assert all(torch.isfinite(g).all() for pair in term_grads.values() for gs in pair for g in gs)
    r = m.retrieve(X,exclude_identical=False,query_case_ids=ids)
    if task == "regression":
        assert torch.allclose(T["L_delta"],T["L_post"],rtol=1e-6,atol=1e-7)
        for x,z in zip(sum(term_grads["L_delta"],[]),sum(term_grads["L_post"],[])):
            assert torch.allclose(x,z,rtol=1e-5,atol=2e-6)
        residual_identity = "L_delta equals L_post in value and both component gradients"
    else:
        p0 = r["weights"]@m.labels[r["case_indices"]].float()
        p0 = p0/p0.sum(1,keepdim=True).clamp_min(1e-12)
        _,scores = a(r["query_features"]-r["weights"]@r["case_features"],p0,None)
        brier = F.mse_loss(scores,F.one_hot(y,2).float())
        assert torch.allclose(T["L_delta"],brier,rtol=1e-6,atol=1e-7)
        for x,z in zip(gradients(brier,rp+ap),sum(term_grads["L_delta"],[])):
            assert torch.allclose(x,z,rtol=1e-5,atol=2e-6)
        residual_identity = "L_delta equals squared error of nominal scores against one-hot labels"
    checks = {}
    for mode in ["summed","separated"]:
        c = dataclasses.replace(cfg,joint_objective=mode)
        V = terms(); R,A,total = objectives(V,c)
        expected = gradients(total,rp+ap) if mode=="summed" else gradients(R,rp)+gradients(A,ap)
        for p in rp+ap: p.grad=None
        joint_backward(V,c,rp,ap)
        actual = [torch.zeros_like(p) if p.grad is None else p.grad for p in rp+ap]
        assert all(torch.allclose(g,h,rtol=1e-5,atol=2e-6) for g,h in zip(actual,expected))
        checks[mode] = "backward matches independently assembled autograd objective"
    saved = [p.detach().clone() for p in rp+ap]
    base_support = support()
    gpost_r = term_grads["L_post"][0]
    direction = [g/norm(gpost_r) for g in gpost_r]
    trials=[]
    for eps in [1e-3,3e-4,1e-4,3e-5]:
        perturbed(rp,saved[:len(rp)],direction,eps)
        plus=float(terms()["L_post"].detach()); same_plus=torch.equal(support(),base_support)
        perturbed(rp,saved[:len(rp)],direction,-eps)
        minus=float(terms()["L_post"].detach()); same_minus=torch.equal(support(),base_support)
        numerical=(plus-minus)/(2*eps); analytical=norm(gpost_r)
        rel=abs(numerical-analytical)/max(abs(analytical),1e-10)
        trials.append(dict(epsilon=eps,numerical=numerical,analytical=analytical,relative_error=rel,
                           mask_support_unchanged=same_plus and same_minus))
        if same_plus and same_minus and rel<.03: break
    perturbed(rp+ap,saved,[torch.zeros_like(p) for p in rp+ap],0.)
    assert any(x["mask_support_unchanged"] and x["relative_error"]<.03 for x in trials),trials
    base_terms={k:float(v.detach()) for k,v in terms().items()}
    descent={}
    for component in ["retrieval_only_on_post","adapter_only_on_post","both_on_total"]:
        V=terms(); total=objectives(V,cfg)[2]
        if component=="retrieval_only_on_post":
            gs=gradients(V["L_post"],rp)+[torch.zeros_like(p) for p in ap]
        elif component=="adapter_only_on_post":
            gs=[torch.zeros_like(p) for p in rp]+gradients(V["L_post"],ap)
        else: gs=gradients(total,rp+ap)
        unit=[g/norm(gs) for g in gs]
        before=float(total.detach()) if component=="both_on_total" else base_terms["L_post"]
        attempts=[]
        for step in [1e-3,1e-4,1e-5,1e-6]:
            perturbed(rp+ap,saved,unit,-step)
            after=terms()
            value=float(objectives(after,cfg)[2].detach()) if component=="both_on_total" else float(after["L_post"].detach())
            attempts.append(dict(step_norm=step,target_loss_before=before,target_loss_after=value,
                                 retrieval_loss_before=base_terms["L_pre"],retrieval_loss_after=float(after["L_pre"].detach()),
                                 same_mask_support=torch.equal(support(),base_support)))
            if value<before: break
        perturbed(rp+ap,saved,[torch.zeros_like(p) for p in rp+ap],0.)
        assert attempts[-1]["target_loss_after"]<before,attempts
        if component=="adapter_only_on_post":
            assert attempts[-1]["retrieval_loss_after"]==base_terms["L_pre"]
        descent[component]=attempts
    # Verify actual optimizer parameter-group coverage and one fresh-Adam update.
    ropt=make_optimizer(m,cc)
    aopt=adapter_optimizer(ap,1e-3,0.,m.cases.device)
    assert {id(p) for g in ropt.param_groups for p in g['params']}=={id(p) for p in rp}
    ropt.zero_grad();aopt.zero_grad()
    adam_before=terms()
    adam_before_values={k:float(adam_before[k].detach()) for k in ['L_pre','L_post']}
    adam_before_values['total']=float(objectives(adam_before,cfg)[2].detach())
    joint_backward(adam_before,cfg,rp,ap)
    torch.nn.utils.clip_grad_norm_(rp,.5)
    ropt.step();aopt.step()
    optimizer_changed = {"retrieval":any(not torch.equal(p,s) for p,s in zip(rp,saved[:len(rp)])),
                         "adapter":any(not torch.equal(p,s) for p,s in zip(ap,saved[len(rp):]))}
    assert all(optimizer_changed.values())
    adam_after=terms()
    adam_after_values={k:float(adam_after[k].detach()) for k in ['L_pre','L_post']}
    adam_after_values['total']=float(objectives(adam_after,cfg)[2].detach())
    result=dict(task=task,representation=representation,mcb=mcb,masked=masked,
                graph={"weights_require_grad":r['weights'].requires_grad,"query_features_require_grad":r['query_features'].requires_grad,
                       "case_features_require_grad":r['case_features'].requires_grad},
                term_grad_norms={k:{"retrieval":norm(v[0]),"adapter":norm(v[1])} for k,v in term_grads.items()},
                per_retrieval_parameter_post_gradient={n:float(g.norm()) for (n,_),g in zip(named_r,gpost_r)},
                retrieval_pre_post_gradient_cosine=cosine(term_grads['L_pre'][0],gpost_r),
                residual_identity=residual_identity,backward_checks=checks,finite_difference_trials=trials,
                descent=descent,actual_adam_changed_parameters=optimizer_changed,
                actual_adam_loss_before=adam_before_values,actual_adam_loss_after=adam_after_values)
    return result


def missing_class_probe():
    X=torch.tensor([[-3.],[-2.],[-1.],[1.],[2.],[3.]])
    y=torch.tensor([0,0,0,1,1,1]);q=torch.tensor([[-2.2]]);target=torch.tensor([1])
    rows=[]
    for masked in [True,False]:
        cc=CoreConfig(task_type='classification',top_k=2,bias_init_k=2,pre_topk_mask=masked,seed=17)
        m=build_model(X,y,cc,2);m.train();a=adapter_for(m,'classification')
        cfg=SyncConfig('joint_rr',1.,.5,0.,1.,1.,0.,score_scale=10.)
        T=_forward_terms(m,a,q,target,None,output_mode=cfg.output_mode,near_scale=1.,score_scale=10.)
        rp=[p for p in m.parameters() if p.requires_grad]
        norm_pre=norm(gradients(T['L_pre'],rp)); norm_post=norm(gradients(T['L_post'],rp))
        rows.append(dict(masked=masked,top_k=2,pre_loss=float(T['L_pre'].detach()),
                         retrieval_gradient_from_pre=norm_pre,retrieval_gradient_from_post=norm_post))
    assert rows[0]['retrieval_gradient_from_pre']==0 and rows[1]['retrieval_gradient_from_pre']>0
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='results/t1/p4_component_sync/gradient_graph_audit_20261030')
    args=parser.parse_args();out=Path(args.out)
    if (out/'audit.json').exists(): raise SystemExit('Choose a fresh --out; preserve prior evidence.')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    write_json(out/'manifest.json',dict(git=git_state(ROOT),source_sha256=source_fingerprint(ROOT),
                                       environment=environment_info(),scope='synthetic local derivatives, not convergence or generalization'))
    (out/'runner_at_execution.py').write_bytes(Path(__file__).read_bytes())
    rows=[]
    for task in ['classification','regression']:
        for representation,mcb in [('raw',False),('mlp',False),('mlp',True)]:
            for masked in [True,False]:
                r=probe(task,representation,masked,mcb);rows.append(r)
                print(task,representation,'mcb=',mcb,'mask=',masked,'post->R gradient=',r['term_grad_norms']['L_post']['retrieval'],flush=True)
    write_json(out/'audit.json',dict(probes=rows,missing_true_class=missing_class_probe(),
                                   note='Descent probes use normalized gradient steps with predefined backtracking; actual Adam is checked for updates, not monotonic loss.'))
    print(f'All {len(rows)} graph probes passed: {out / "audit.json"}',flush=True)


if __name__=='__main__': main()
