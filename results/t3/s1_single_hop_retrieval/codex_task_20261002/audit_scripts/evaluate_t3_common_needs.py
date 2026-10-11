import hashlib
import json
from pathlib import Path
import sys

import torch
work=Path(__file__).resolve().parent;outputs=work.parent/'outputs'
freeze=json.loads((work/'t3_train_need_queries_freeze.json').read_text());archive=Path(freeze['archive']);sys.path.insert(0,str(archive))
from model.t3.benchmarks import load_squad
from model.t3.lexical import candidate_bank,core_bank,hash_vector,need_loss,bm25_rank
from model.t3.need_queries import load_need_queries
from model.nnknn_model import GlocalFeatureWeight
verified=json.loads((outputs/'neural-cbr-t3-train-need-query-verification.json').read_text())
assert verified['all_recorded_checks_passed'] and verified['queries']==192
for name,bound in verified['input_artifacts'].items():assert hashlib.sha256(Path(bound['path']).read_bytes()).hexdigest()==bound['sha256']
samples_path=outputs/'t3-public-data-audit-20261005-preserved-gaps/verification.json'
samples=json.loads(samples_path.read_text())['samples']['squad_train_v11.json']
source=json.loads((outputs/'t3-train-need-queries-20261005/need_queries.json').read_text())['protocol']
tasks,_=load_squad(work/'t3-public-data-20261005/squad_train_v11.json',expected_sha256=source['data_sha256'])
by_id={t.question_id:t for t in tasks};groups={name:[by_id[qid] for qid in samples[name+'_ids']] for name in ('fit','tune')}
selected=groups['fit']+groups['tune']
actual,binding=load_need_queries(outputs/'t3-train-need-queries-20261005/need_queries.json',selected,data_sha256=source['data_sha256'],sample_sha256=hashlib.sha256(samples_path.read_bytes()).hexdigest())
query_sets={'original_question':{t.question_id:t.question for t in selected},'actual_need':actual}
corpus=tuple({c.case_id:c for t in tasks for c in t.cases}.values())
banks={t.question_id:candidate_bank(t,corpus,size=64) for t in selected}
torch.set_num_threads(1)
metric=GlocalFeatureWeight(256,1)
models={qid:core_bank(bank,metric=metric).eval().requires_grad_(False) for qid,bank in banks.items()}
conditions=[('fixed',torch.ones(1,256),None)]
for package in ('question-control','actual'):
    for seed in (8,9,10):
        path=outputs/f't3-train-need-{package}-20261005/s{seed}/metric.pt'
        checkpoint=torch.load(path,weights_only=True);conditions.append((f'{package}/s{seed}',checkpoint['selected_metric']['feature_weights'],dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
records=[];manual_candidates=0;nn_calls=0;lexical_calls=0
for query_name,queries in query_sets.items():
    for condition,weights,checkpoint in conditions:
        metric.feature_weights.data.copy_(weights)
        for group_name,group in groups.items():
            total=hits1=hits2=0;events=[]
            for task in group:
                model=models[task.question_id];query=queries[task.question_id]
                positive_ids=[c.case_id for c in task.cases]
                with torch.no_grad():loss,result=need_loss(model,query,positive_ids,objective='hard_negative')
                contrib=(model.cases-hash_vector(query)).square()*weights[0]
                dist=contrib.sum(-1).sqrt()
                torch.testing.assert_close(result['distances'][0],dist,rtol=0,atol=0)
                torch.testing.assert_close(result['feature_distance_contributions'][0],contrib,rtol=0,atol=0)
                pos=torch.isin(model.active_case_ids(),torch.tensor(positive_ids))
                manual=torch.nn.functional.softplus((dist[pos].min()-dist[~pos].sort(stable=True).values[:8]+.2)/model.tau).mean()
                torch.testing.assert_close(loss,manual,rtol=0,atol=0)
                order=torch.argsort(dist,stable=True);chosen=model.active_case_ids()[order[:2]].tolist()
                hits1+=chosen[0] in positive_ids;hits2+=bool(set(chosen)&set(positive_ids));total+=float(loss)
                events.append(dict(question_id=task.question_id,need=query,positive_ids=positive_ids,selected_ids=chosen,loss=float(loss)))
                nn_calls+=1;manual_candidates+=len(banks[task.question_id])
            records.append(dict(query_source=query_name,condition=condition,group=group_name,questions=len(group),mean_need_loss=total/len(group),source_recall_at1=hits1/len(group),source_recall_at2=hits2/len(group),checkpoint=checkpoint,events=events))
    for group_name,group in groups.items():
        h1=h2=0
        for task in group:
            ranked=bm25_rank(queries[task.question_id],banks[task.question_id]);positive={c.case_id for c in task.cases}
            h1+=ranked[0]['case_id'] in positive;h2+=bool({r['case_id'] for r in ranked[:2]}&positive);lexical_calls+=1
        records.append(dict(query_source=query_name,condition='bm25',group=group_name,questions=len(group),source_recall_at1=h1/len(group),source_recall_at2=h2/len(group)))
assert nn_calls==2688 and lexical_calls==384 and manual_candidates==172032
report=dict(source=freeze,need_binding=binding,counts=dict(actual_nn_evaluations=nn_calls,actual_bm25_rankings=lexical_calls,manual_candidates=manual_candidates),records=records,all_recorded_checks_passed=True,
    scope='same prespecified article-disjoint fit/tune questions and candidate banks; every selected metric evaluated on both common actual needs and original questions; independent geometry/loss; tune remains selection data,not confirmation; no new LLM generation or outcome utility claim')
(outputs/'neural-cbr-t3-common-need-evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps([dict(query_source=r['query_source'],condition=r['condition'],recall2=r['source_recall_at2']) for r in records if r['group']=='tune']))
