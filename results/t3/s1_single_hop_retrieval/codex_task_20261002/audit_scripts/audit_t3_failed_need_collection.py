import hashlib
import json
from pathlib import Path
import sys
import zipfile
work=Path(__file__).resolve().parent
freeze=json.loads((work/'t3_train_need_queries_freeze.json').read_text())
archive=Path(freeze['archive']);sys.path.insert(0,str(archive))
from model.t3.host_schema import request_prompt,validate_decision
from model.t3.need_queries import load_need_queries
out=work.parent/'outputs/t3-train-need-queries-20261005'
artifact=json.loads((out/'need_queries.json').read_text());protocol=artifact['protocol']
assert protocol['source_fingerprint']==freeze['fingerprint']
assert hashlib.sha256(Path(freeze['zip']).read_bytes()).hexdigest()==freeze['zip_sha256']
with zipfile.ZipFile(freeze['zip']) as bundle:
    assert bundle.testzip() is None and len(bundle.namelist())==147
    digest=hashlib.sha256()
    for name in bundle.namelist():
        data=bundle.read(name);assert data==(archive/name).read_bytes()
        digest.update(name.encode());digest.update(data)
    assert digest.hexdigest()==freeze['fingerprint']
summary=json.loads((out/'summary.json').read_text())
records=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
calls=[json.loads(line) for line in (out/'host_calls.jsonl').read_text().splitlines()]
assert records==artifact['records'] and len(records)==len(calls)==summary['calls']==192
samples=json.loads((work.parent/'outputs/t3-public-data-audit-20261005-preserved-gaps/verification.json').read_text())['samples']['squad_train_v11.json']
assert [row['question_id'] for row in records]==samples['fit_ids']+samples['tune_ids']
errors=[]
for index,(record,stored) in enumerate(zip(records,calls),1):
    call=record['call']
    assert call==stored['call'] and record['question_id']==stored['question_id']
    assert call['stage']=='request' and call['source_prompt']==request_prompt(record['question'])
    assert call['source_prompt'] in call['prompt'] and 1<=call['output_tokens']<=128
    try:decision=validate_decision(json.loads(call['output'].strip()),'request')
    except (ValueError,KeyError,TypeError) as exc:
        assert record['request'] is None and record['error']==str(exc)
        errors.append(dict(position=index,question_id=record['question_id'],error=record['error'],output_tokens=call['output_tokens']))
    else:assert record['error'] is None and decision==record['request']
assert len(errors)==summary['errors'] and 192-len(errors)==summary['valid_requests']
assert summary['input_tokens']==sum(row['call']['input_tokens'] for row in records)
assert summary['output_tokens']==sum(row['call']['output_tokens'] for row in records)
assert summary['generation_seconds']==sum(row['call']['seconds'] for row in records)
from types import SimpleNamespace
tasks=[SimpleNamespace(question_id=row['question_id'],question=row['question']) for row in records]
try:load_need_queries(out/'need_queries.json',tasks,data_sha256=protocol['data_sha256'],sample_sha256=protocol['sample_sha256'])
except ValueError as exc:assert 'unobserved' in str(exc)
else:raise AssertionError('Incomplete collection incorrectly accepted')
assert not (work.parent/'outputs/t3-train-need-actual-20261005/summary.json').exists()
report=dict(source=freeze,summary=summary,errors=errors,all_recorded_checks_passed=True,
    incomplete_fitting_rejected=True,input_artifacts={name:dict(path=str((out/name).resolve()),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in ('protocol.json','need_queries.json','records.jsonl','host_calls.jsonl','summary.json')},
    boundary='complete failed collection accounting; not successful need coverage, tokenizer/model replay, semantic relevance or trained metric')
(work.parent/'outputs/neural-cbr-t3-failed-need-collection-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(summary=summary,errors=errors,incomplete_fitting_rejected=True)))
