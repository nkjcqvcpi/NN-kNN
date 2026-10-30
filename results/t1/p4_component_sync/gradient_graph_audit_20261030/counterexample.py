"""Constructed zero-retrieval-loss example using the actual T1 regression graph."""
import json
import os
from pathlib import Path
import sys
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[3]))
os.environ['NNKNN_DEVICE']='cpu'
import torch
from model.common.core import CoreConfig,build_model
from model.nn_cdh import NNCDHAdapter
from model.t1.sync import SyncConfig,_forward_terms

torch.set_num_threads(1)
X=torch.tensor([[-3.],[-2.],[-1.],[1.],[2.],[3.]])
y=torch.tensor([-2.,-1.,-.5,.5,1.,2.])
q=torch.tensor([[-1.4],[.2],[1.6]])
m=build_model(X,y,CoreConfig(task_type='regression',pre_topk_mask=False,bias_init_k=2,seed=17),None)
m.train()
a=NNCDHAdapter(1,1,(8,4));a.adapt_net_pair.requires_grad_(False)
with torch.no_grad():
    for p in a.adapt_net_final.parameters(): p.zero_()
    a.adapt_net_final.model[-1].bias.fill_(1.)
    r=m.retrieve(q,exclude_identical=False)
    target=(r['weights']@m.labels[r['case_indices']]).detach().view(-1)
cfg=SyncConfig('joint_rr',1.,.5,0.,1.,1.,0.,output_mode='aggregate_regression',joint_objective='summed')
def terms():
    return _forward_terms(m,a,q,target,None,output_mode='aggregate_regression',near_scale=1.)
def total(T): return T['L_pre']+1.5*T['L_post']+T['L_delta']
T=terms();R=[p for p in m.parameters() if p.requires_grad];A=[p for p in a.parameters() if p.requires_grad]
gpre=torch.autograd.grad(T['L_pre'],R,retain_graph=True,allow_unused=True)
assert all(g is None or torch.count_nonzero(g)==0 for g in gpre)
gs=torch.autograd.grad(total(T),R+A,allow_unused=True)
gs=[torch.zeros_like(p) if g is None else g for p,g in zip(R+A,gs)]
length=sum(float(g.double().square().sum()) for g in gs)**.5
assert any(torch.count_nonzero(g)>0 for g in gs[:len(R)])
before={k:float(T[k].detach()) for k in ['L_pre','L_post','L_delta']}
before['total']=float(total(T).detach())
with torch.no_grad():
    for p,g in zip(R+A,gs): p.add_(g,alpha=-1e-3/length)
V=terms();after={k:float(V[k].detach()) for k in ['L_pre','L_post','L_delta']};after['total']=float(total(V).detach())
assert before['L_pre']==0 and after['L_pre']>0 and after['total']<before['total']
result=dict(scope='Constructed example: query targets are fixed to the initial retrieval predictions; adapter starts with correction +1. Not a benchmark result.',
            objective='L_pre + 2.5 * L_post; L_small inactive',step='normalized negative total gradient, parameter displacement norm 0.001',
            retrieval_gradient_from_pre_zero=True,retrieval_gradient_from_total_nonzero=True,before=before,after=after)
(OUT/'counterexample.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
