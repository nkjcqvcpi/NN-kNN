import hashlib
import json
from pathlib import Path
import sys
work=Path(__file__).resolve().parent
freeze=json.loads((work/'t3_bounded_request_256_freeze.json').read_text());sys.path.insert(0,freeze['archive'])
from model.t3.lexical import hash_vector
import torch
root=work.parent/'outputs';out=root/'t3-train-need-queries-bounded-20261006'
audit=json.loads((root/'neural-cbr-t3-train-need-query-bounded-verification.json').read_text())
assert audit['all_recorded_checks_passed'] and audit['queries']==192
for name,bound in audit['input_artifacts'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==bound['sha256']
artifact=json.loads((out/'need_queries.json').read_text());records=artifact['records']
old=json.loads((root/'t3-train-need-queries-256-20261005/need_queries.json').read_text())
assert len(records)==len(old['records'])==192
same_good=0;repairs=[];groups={}
for group,ids in [('fit',artifact['protocol']['fit_ids']),('tune',artifact['protocol']['tune_ids'])]:
    group_rows=[row for row in records if row['question_id'] in ids]
    groups[group]=dict(queries=len(group_rows),exact_question_needs=sum(r['request']['need']==r['question'] for r in group_rows),
        hash_vector_changed=sum(not torch.equal(hash_vector(r['question']),hash_vector(r['request']['need'])) for r in group_rows),
        maximum_need_characters=max(len(r['request']['need']) for r in group_rows))
for index,(new,previous) in enumerate(zip(records,old['records']),1):
    assert new['question_id']==previous['question_id'] and new['error'] is None
    if previous['error'] is None:
        assert new['call']['output']==previous['call']['output'];same_good+=1
    else:repairs.append(dict(position=index,question_id=new['question_id'],need_characters=len(new['request']['need']),output_tokens=new['call']['output_tokens']))
assert same_good==190 and len(repairs)==2
summaries={name:json.loads((root/name/'summary.json').read_text()) for name in ('t3-train-need-queries-20261005','t3-train-need-queries-256-20261005','t3-train-need-queries-bounded-20261006')}
report=dict(source=freeze,summary=summaries,unchanged_successful_outputs=190,repaired_requests=repairs,query_shift=groups,
    verified_inputs=audit['input_artifacts'],all_recorded_checks_passed=True,
    boundary='complete actual train-only needs; same190 outputs/two string closures; semantic repetition remains; no fitted retrieval or downstream utility claim')
(root/'neural-cbr-t3-complete-need-collection-results.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('source','summary','verified_inputs')}))
