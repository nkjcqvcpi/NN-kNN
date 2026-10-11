import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch

work=Path(__file__).resolve().parent
freeze=json.loads((work/'t3_need_matching_freeze.json').read_text())
archive=Path(freeze['archive']);sys.path.insert(0,str(archive))
from model.t1.artifacts import source_fingerprint
from model.t3.lexical import hash_vector
import tools.t3_need_matching as driver

assert source_fingerprint(archive)==freeze['fingerprint']
assert hashlib.sha256(Path(freeze['zip']).read_bytes()).hexdigest()==freeze['zip_sha256']
with zipfile.ZipFile(freeze['zip']) as bundle:
    assert bundle.testzip() is None and len(bundle.namelist())==138
    assert all(bundle.read(name)==(archive/name).read_bytes() for name in bundle.namelist())
counts=dict(actual_retrainings=3,actual_training_steps=0,evaluation_calls=0,nonzero_gradients=0,max_manual_gradient_error=0.)
real=driver.need_loss;expected_grad=None
def audited(model,question,positive_ids,*,dimensions):
    global expected_grad
    value,result=real(model,question,positive_ids,dimensions=dimensions)
    q=hash_vector(question,dimensions).unsqueeze(0)
    squared=(model.cases-q).square()
    weights=torch.nn.functional.leaky_relu(model.glocal_weightor.feature_weights,negative_slope=.001)[0]
    row_weights=torch.nn.functional.leaky_relu(model.glocal_weights[:,0],negative_slope=.001)
    contrib=squared*weights*row_weights[:,None]
    distances=contrib.sum(-1).clamp_min(0).sqrt()
    torch.testing.assert_close(distances,result['distances'][0],rtol=0,atol=0)
    torch.testing.assert_close(contrib,result['feature_distance_contributions'][0],rtol=0,atol=0)
    logits=(model.biases-distances)/model.tau
    positive=torch.isin(model.active_case_ids(),torch.tensor(positive_ids))
    manual=torch.logsumexp(logits,0)-torch.logsumexp(logits[positive],0)
    torch.testing.assert_close(value,manual,rtol=0,atol=0)
    if torch.is_grad_enabled():
        expected_grad=torch.autograd.grad(manual,model.glocal_weightor.feature_weights)[0]
        counts['actual_training_steps']+=1
    else:counts['evaluation_calls']+=1
    return value,result
driver.need_loss=audited
real_step=torch.optim.Adam.step
def step(optimizer,*args,**kwargs):
    params=[p for group in optimizer.param_groups for p in group['params']]
    assert len(params)==1 and params[0].numel()==256
    counts['max_manual_gradient_error']=max(counts['max_manual_gradient_error'],float((params[0].grad-expected_grad).abs().max()))
    torch.testing.assert_close(params[0].grad,expected_grad,rtol=1e-5,atol=1e-7)
    counts['nonzero_gradients']+=int(bool((expected_grad!=0).any()))
    return real_step(optimizer,*args,**kwargs)
torch.optim.Adam.step=step
output=work/'t3-need-matching-replayed-20261005-gradient-tolerance'
sys.argv=['t3_need_matching','--data-root',str(work/'t3-public-data-20261005'),'--samples',
    str(work.parent/'outputs/t3-public-data-audit-20261005-preserved-gaps/verification.json'),'--output',str(output)]
driver.main()

def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
original=work/'t3-need-matching-20261005'
assert json.loads((original/'protocol.json').read_text())==json.loads((output/'protocol.json').read_text())
for seed in (8,9,10):
    a,b=[torch.load(folder/f's{seed}/metric.pt',weights_only=True,map_location='cpu') for folder in (original,output)]
    assert equal(a,b)
    a,b=[json.loads((folder/f's{seed}/summary.json').read_text()) for folder in (original,output)]
    for report in (a,b):report.pop('training_seconds')
    assert a==b
    assert (original/f's{seed}/training_events.json').read_bytes()==(output/f's{seed}/training_events.json').read_bytes()
assert counts['actual_training_steps']==counts['nonzero_gradients']==7680
report=dict(source=freeze,counts=counts,all_checks_passed=True,
    scope='three actual complete retrainings,manual geometry/loss every call exactly; independent float32 gradient rtol1e-5/atol1e-7; all original selected/final Adam and parameter states exactly match')
(work.parent/'outputs/neural-cbr-t3-need-matching-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(counts))
