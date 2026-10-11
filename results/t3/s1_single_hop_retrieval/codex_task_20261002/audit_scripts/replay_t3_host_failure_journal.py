import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch
work=Path(__file__).resolve().parent
freeze=json.loads((work/'t3_event_journal_freeze.json').read_text())
archive=Path(freeze['archive']);sys.path.insert(0,str(archive))
from model.t1.artifacts import source_fingerprint
from model.t3.lexical import core_bank,HashQuery
from model.t3.retrieval import Case,Retriever,Access
from model.t3.orchestrator import Budget,run_loop
assert source_fingerprint(archive)==freeze['fingerprint']
assert hashlib.sha256(Path(freeze['zip']).read_bytes()).hexdigest()==freeze['zip_sha256']
with zipfile.ZipFile(freeze['zip']) as bundle:
    assert bundle.testzip() is None
    assert all(bundle.read(name)==(archive/name).read_bytes() for name in bundle.namelist())
out=work.parent/'outputs/t3-public-host-20261005-request-v2'
protocol=json.loads((out/'protocol.json').read_text())
rows=[json.loads(line) for line in (out/'trials.jsonl').read_text().splitlines()]
failed=[r for r in rows if 'error' in r and r['condition'] in {'fixed','learned','iterative'} and len(r['host_calls'])>1]
assert failed
metric=torch.load(work/'t3-need-matching-20261005/s8/metric.pt',weights_only=True)['selected_metric']['feature_weights']
records=[]
for row in failed:
    bank=[Case(**c) for c in protocol['case_pools'][row['dataset']][row['question_id']]]
    model=core_bank(bank)
    if row['condition']!='fixed':model.glocal_weightor.feature_weights.data.copy_(metric)
    model.eval().requires_grad_(False)
    retriever=Retriever(model,bank,HashQuery(),model_version='journal-replay',encoder_version='hash256-v1')
    journal=[];index=0
    def host(**kwargs):
        global index
        call=row['host_calls'][index]
        if index:
            assert json.dumps(kwargs['evidence'],ensure_ascii=False) in call['prompt']
        index+=1
        raw=call['output'].strip()
        if raw.startswith('```json') and raw.endswith('```'):raw=raw[7:-3].strip()
        return json.loads(raw)
    try:
        run_loop('archived public question',host,retriever,Access('public-benchmark','public-benchmark'),
            Budget(max_rounds=2 if row['condition']=='iterative' else 1,max_cases=2,max_evidence_chars=12000,max_seconds=240,
                   max_cases_per_round=1 if row['condition']=='iterative' else 2),now=0,event_sink=journal.append)
    except (ValueError,KeyError,TypeError) as exc:
        assert str(exc)==row['error'],(str(exc),row['error'])
    else:raise AssertionError('archived malformed host output must still fail')
    assert journal and all(e['delivered_to_host'] for e in journal)
    assert all(e['event_id']==e['audit']['event_id'] for e in journal)
    records.append(dict(dataset=row['dataset'],question_id=row['question_id'],condition=row['condition'],
                        archived_error=row['error'],new_actual_retrieval_events=journal,preserved_before_error=True))
report=dict(source=freeze,input_prefix_rows=len(rows),input_prefix_canonical_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
    recorded_failed_trials=len(records),preserved_events=sum(len(r['new_actual_retrieval_events']) for r in records),
    records=records,all_checks_passed=True,scope='new real core retrievals against archived real malformed host outputs,matching delivered prompt payloads; not fresh LLM generations or recovery of lost original event IDs')
(work.parent/'outputs/neural-cbr-t3-failure-journal-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('failure journal actual replays',len(records),report['preserved_events'])
