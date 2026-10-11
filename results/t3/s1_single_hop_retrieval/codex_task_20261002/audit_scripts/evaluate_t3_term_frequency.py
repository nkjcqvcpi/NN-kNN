from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import torch
work=Path(__file__).resolve().parent;root=work.parent/'outputs'
freeze=json.loads((work/'t3_term_frequency_freeze.json').read_text());sys.path.insert(0,freeze['archive'])
from model.t3.benchmarks import load_squad
from model.t3.lexical import core_bank,need_loss,bm25_rank
from model.nnknn_model import GlocalFeatureWeight
verified=json.loads((root/'neural-cbr-t3-term-frequency-retraining-verification.json').read_text())
assert verified['all_recorded_checks_passed'] and verified['counts']['actual_retrainings']==6
artifact=json.loads((root/'t3-train-need-queries-bounded-20261006/need_queries.json').read_text())
source=artifact['protocol']
tasks,_=load_squad(work/'t3-public-data-20261005/squad_train_v11.json',expected_sha256=source['data_sha256'])
by_id={t.question_id:t for t in tasks};cases={c.case_id:c for t in tasks for c in t.cases}
original=json.loads((root/'t3-train-need-actual-bounded-20261006/protocol.json').read_text())
banks={qid:tuple(cases[i] for i in ids) for qid,ids in original['candidate_ids'].items()}
groups={group:[by_id[qid] for qid in source[group+'_ids']] for group in ('fit','tune')}
actual={r['question_id']:r['request']['need'] for r in artifact['records']};assert len(actual)==192
queries={'original_question':{t.question_id:t.question for group in groups.values() for t in group},'actual_need':actual}
def f32(value):return struct.unpack('f',struct.pack('f',value))[0]
def encode(text,mode):
    values=[0.]*256
    for token,count in Counter(re.findall('[a-z0-9]+',text.lower())).items():
        index=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')%256
        value=count if mode=='count' else 1+math.log(count) if mode=='sublinear' else 1.
        values[index]=f32(values[index]+f32(value))
    vector=torch.tensor(values);return vector/vector.norm().clamp_min(1e-9)
def text(case):
    obj=json.loads(case.content);return obj['title']+' '+(obj['passage'] if 'passage' in obj else ' '.join(obj['sentences']))
torch.set_num_threads(1);records=[];calls=manual=0
for mode in ('count','sublinear','binary'):
    metric=GlocalFeatureWeight(256,1)
    models={qid:core_bank(bank,metric=metric,term_frequency=mode).eval().requires_grad_(False) for qid,bank in banks.items()}
    cached={i:encode(text(cases[i]),mode) for ids in original['candidate_ids'].values() for i in ids}
    features={qid:torch.stack([cached[i] for i in ids]) for qid,ids in original['candidate_ids'].items()}
    conditions=[('fixed',torch.ones(1,256),None)]
    for seed in (8,9,10):
        package='t3-train-need-actual-bounded-20261006' if mode=='count' else f't3-need-{mode}-20261006'
        path=root/package/f's{seed}/metric.pt';weights=torch.load(path,weights_only=True)['selected_metric']['feature_weights']
        conditions.append((f's{seed}',weights,dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    for name,weights,binding in conditions:
        metric.feature_weights.data.copy_(weights)
        for query_name,texts in queries.items():
            for group_name,group in groups.items():
                total=h1=h2=0;events=[]
                for task in group:
                    model=models[task.question_id];query=texts[task.question_id];positive=[c.case_id for c in task.cases]
                    with torch.no_grad():loss,result=need_loss(model,query,positive,objective='hard_negative',term_frequency=mode)
                    X=features[task.question_id];torch.testing.assert_close(X,model.cases,rtol=0,atol=0)
                    contrib=(X-encode(query,mode)).square()*weights[0];dist=contrib.sum(-1).sqrt()
                    torch.testing.assert_close(result['feature_distance_contributions'][0],contrib,rtol=0,atol=0)
                    torch.testing.assert_close(result['distances'][0],dist,rtol=0,atol=0)
                    mask=torch.isin(model.active_case_ids(),torch.tensor(positive))
                    expected=torch.nn.functional.softplus((dist[mask].min()-dist[~mask].sort(stable=True).values[:8]+.2)/model.tau).mean()
                    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
                    order=torch.argsort(dist,stable=True);selected=model.active_case_ids()[order[:2]].tolist()
                    h1+=selected[0] in positive;h2+=bool(set(selected)&set(positive));total+=float(loss)
                    events.append(dict(question_id=task.question_id,need=query,positive_ids=positive,selected_ids=selected))
                    calls+=1;manual+=len(X)
                records.append(dict(mode=mode,condition=name,query_source=query_name,group=group_name,questions=len(group),
                    mean_need_loss=total/len(group),source_recall_at1=h1/len(group),source_recall_at2=h2/len(group),checkpoint=binding,events=events))
lexical=0
for query_name,texts in queries.items():
    for group_name,group in groups.items():
        h1=h2=0
        for task in group:
            rank=bm25_rank(texts[task.question_id],banks[task.question_id]);positive={c.case_id for c in task.cases}
            h1+=rank[0]['case_id'] in positive;h2+=bool({r['case_id'] for r in rank[:2]}&positive);lexical+=1
        records.append(dict(mode='bm25',condition='fixed',query_source=query_name,group=group_name,questions=len(group),source_recall_at1=h1/len(group),source_recall_at2=h2/len(group)))
assert calls==4608 and manual==294912 and lexical==384
report=dict(source=freeze,counts=dict(actual_nn_evaluations=calls,manual_candidates=manual,actual_bm25_rankings=lexical),records=records,
    all_recorded_checks_passed=True,scope='all12 fixed/selected encodings on both same actual needs and original questions/fixed banks; independent features/geometry/loss; tune is selection data; no host utility claim')
(root/'neural-cbr-t3-term-frequency-common-evaluation.json').write_text(json.dumps(report,indent=2))
print(json.dumps([{key:r[key] for key in ('mode','condition','source_recall_at1','source_recall_at2')} for r in records if r['group']=='tune' and r['query_source']=='actual_need'],indent=2))
