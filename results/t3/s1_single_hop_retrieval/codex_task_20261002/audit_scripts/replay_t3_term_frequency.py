from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import math
import zipfile
import torch
work=Path(__file__).resolve().parent;root=work.parent/'outputs'
freeze=json.loads((work/'t3_term_frequency_freeze.json').read_text());archive=Path(freeze['archive']);sys.path.insert(0,str(archive))
from model.t1.artifacts import source_fingerprint
from model.t3.benchmarks import load_squad
import tools.t3_need_matching as driver
assert source_fingerprint(archive)==freeze['fingerprint']
assert hashlib.sha256(Path(freeze['zip']).read_bytes()).hexdigest()==freeze['zip_sha256']
with zipfile.ZipFile(freeze['zip']) as bundle:
    assert bundle.testzip() is None and len(bundle.namelist())==153
    assert all(bundle.read(name)==(archive/name).read_bytes() for name in bundle.namelist())
artifact=json.loads((root/'t3-train-need-queries-bounded-20261006/need_queries.json').read_text())
tasks,_=load_squad(work/'t3-public-data-20261005/squad_train_v11.json',expected_sha256=artifact['protocol']['data_sha256'])
cases={c.case_id:c for t in tasks for c in t.cases}
def f32(value):return struct.unpack('f',struct.pack('f',value))[0]
def encode(text,mode):
    values=[0.]*256
    for token,count in Counter(re.findall('[a-z0-9]+',text.lower())).items():
        value=1+math.log(count) if mode=='sublinear' else 1.
        index=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')%256
        values[index]=f32(values[index]+f32(value))
    vector=torch.tensor(values)
    return vector/vector.norm().clamp_min(1e-9)
counts=dict(actual_retrainings=6,actual_training_steps=0,evaluation_calls=0,nonzero_gradients=0,max_manual_gradient_error=0.,query_checks=0,case_vector_checks=0)
real_loss=driver.need_loss;real_step=torch.optim.Adam.step
mode=None;lookup={};queries={};features={};expected_grad=None
def audited(model,question,positive_ids,*,dimensions,objective,term_frequency):
    global expected_grad
    assert dimensions==256 and term_frequency==mode
    qid=lookup[tuple(model.active_case_ids().tolist())]
    assert question==queries[qid];counts['query_checks']+=1
    value,result=real_loss(model,question,positive_ids,dimensions=dimensions,objective=objective,term_frequency=term_frequency)
    torch.testing.assert_close(model.cases,features[qid],rtol=0,atol=0);counts['case_vector_checks']+=len(model.cases)
    q=encode(question,mode).unsqueeze(0)
    contrib=(features[qid]-q).square()*model.glocal_weightor.feature_weights[0]
    distances=contrib.sum(-1).sqrt()
    torch.testing.assert_close(result['feature_distance_contributions'][0],contrib,rtol=0,atol=0)
    torch.testing.assert_close(result['distances'][0],distances,rtol=0,atol=0)
    pos=torch.isin(model.active_case_ids(),torch.tensor(positive_ids))
    manual=torch.nn.functional.softplus((distances[pos].min()-distances[~pos].sort(stable=True).values[:8]+.2)/model.tau).mean()
    torch.testing.assert_close(value,manual,rtol=0,atol=0)
    if torch.is_grad_enabled():expected_grad=torch.autograd.grad(manual,model.glocal_weightor.feature_weights)[0];counts['actual_training_steps']+=1
    else:counts['evaluation_calls']+=1
    return value,result
def step(optimizer,*args,**kwargs):
    params=[p for g in optimizer.param_groups for p in g['params']];assert len(params)==1 and params[0].numel()==256
    error=float((params[0].grad-expected_grad).abs().max());counts['max_manual_gradient_error']=max(counts['max_manual_gradient_error'],error)
    torch.testing.assert_close(params[0].grad,expected_grad,rtol=1e-5,atol=1e-7)
    counts['nonzero_gradients']+=int(bool((expected_grad!=0).any()))
    return real_step(optimizer,*args,**kwargs)
driver.need_loss=audited;torch.optim.Adam.step=step
def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
runs=[]
for mode in ('sublinear','binary'):
    original=root/f't3-need-{mode}-20261006';output=work/f't3-need-{mode}-20261006-replayed'
    protocol=json.loads((original/'protocol.json').read_text());queries=protocol['query_texts']
    lookup={tuple(ids):qid for qid,ids in protocol['candidate_ids'].items()};assert len(lookup)==192
    def text(case):
        obj=json.loads(case.content);return obj['title']+' '+(obj['passage'] if 'passage' in obj else ' '.join(obj['sentences']))
    cached={case_id:encode(text(cases[case_id]),mode) for ids in protocol['candidate_ids'].values() for case_id in ids}
    features={qid:torch.stack([cached[i] for i in ids]) for qid,ids in protocol['candidate_ids'].items()}
    sys.argv=['fit','--data-root',str(work/'t3-public-data-20261005'),'--samples',str(root/'t3-public-data-audit-20261005-preserved-gaps/verification.json'),
        '--need-queries',str(root/'t3-train-need-queries-bounded-20261006/need_queries.json'),'--output',str(output),'--objective','hard_negative','--selection','tune_recall','--term-frequency',mode]
    driver.main()
    assert protocol==json.loads((output/'protocol.json').read_text())
    for seed in (8,9,10):
        assert equal(torch.load(original/f's{seed}/metric.pt',weights_only=True),torch.load(output/f's{seed}/metric.pt',weights_only=True))
        a,b=[json.loads((folder/f's{seed}/summary.json').read_text()) for folder in (original,output)]
        for row in (a,b):row.pop('training_seconds')
        assert a==b and (original/f's{seed}/training_events.json').read_bytes()==(output/f's{seed}/training_events.json').read_bytes()
        runs.append(dict(mode=mode,seed=seed,selected_epoch=a['selected_epoch'],selected=a['selected']))
assert counts['actual_training_steps']==counts['nonzero_gradients']==15360
assert counts['query_checks']==counts['actual_training_steps']+counts['evaluation_calls']
report=dict(source=freeze,counts=counts,runs=runs,all_recorded_checks_passed=True,
    scope='six actual full retrainings; independent per-word float32 features/geometry/loss/gradients; exact selected/final/Adam/events/curves; no host output or utility claim')
(root/'neural-cbr-t3-term-frequency-retraining-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(counts))
