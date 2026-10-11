import hashlib
import json
from pathlib import Path
import sys
work=Path(__file__).resolve().parent;root=work.parent/'outputs'
freeze=json.loads((work/'t3_bounded_request_256_freeze.json').read_text());sys.path.insert(0,freeze['archive'])
from model.t3.benchmarks import load_squad
from model.t3.lexical import bm25_rank
artifact=json.loads((root/'t3-train-need-queries-bounded-20261006/need_queries.json').read_text())
old=json.loads((root/'t3-train-need-queries-256-20261005/need_queries.json').read_text())
tasks,_=load_squad(work/'t3-public-data-20261005/squad_train_v11.json',expected_sha256=artifact['protocol']['data_sha256'])
by_id={t.question_id:t for t in tasks};cases={c.case_id:c for t in tasks for c in t.cases}
training=json.loads((root/'t3-train-need-actual-bounded-20261006/protocol.json').read_text())
common=json.loads((root/'neural-cbr-t3-common-need-bounded-evaluation.json').read_text())
rows=[]
for record,previous in zip(artifact['records'],old['records']):
    if previous['error'] is None:continue
    qid=record['question_id'];task=by_id[qid];positive={c.case_id for c in task.cases}
    bank=[cases[i] for i in training['candidate_ids'][qid]]
    probes={}
    for query_name,text in [('original_question',task.question),('actual_need',record['request']['need'])]:
        rank=bm25_rank(text,bank)
        fixed=next(row for row in common['records'] if row['query_source']==query_name and row['condition']=='fixed' and row['group']=='tune')
        event=next(row for row in fixed['events'] if row['question_id']==qid)
        probes[query_name]=dict(bm25_source_rank=next(i+1 for i,c in enumerate(rank) if c['case_id'] in positive),
            bm25_selected_ids=[c['case_id'] for c in rank[:2]],fixed_selected_ids=event['selected_ids'],
            fixed_source_recall2=bool(set(event['selected_ids'])&positive))
    rows.append(dict(question_id=qid,question=task.question,actual_need=record['request']['need'],probes=probes))
assert len(rows)==2
report=dict(rows=rows,actual_bm25_rankings=4,scope='same-bank direct-question counterfactual on the two repaired tune needs; source-identity ranks only; no primary query substitution or utility claim')
(root/'neural-cbr-t3-repaired-need-rank-diagnosis.json').write_text(json.dumps(report,indent=2))
print(json.dumps([dict(question_id=row['question_id'],probes=row['probes']) for row in rows],indent=2))
