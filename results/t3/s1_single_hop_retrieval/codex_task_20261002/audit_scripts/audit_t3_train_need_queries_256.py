import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

work=Path(__file__).resolve().parent
native=Path('C:/Users/Administrator/NN-KNN/nnknn-work/.venv/Library/bin')
os.environ['PATH']=str(native)+os.pathsep+os.environ.get('PATH','')
native_handle=os.add_dll_directory(str(native)) if sys.platform=='win32' else None
freeze=json.loads((work/'t3_uniform_request_256_freeze.json').read_text());archive=Path(freeze['archive']);sys.path.insert(0,str(archive))
from model.t1.artifacts import source_fingerprint
from model.t3.benchmarks import load_squad
from model.t3.need_queries import load_need_queries
from model.t3.host_schema import request_prompt,validate_decision
from transformers import AutoTokenizer
out=work.parent/'outputs/t3-train-need-queries-256-20261005'
artifact=json.loads((out/'need_queries.json').read_text());protocol=artifact['protocol']
assert protocol==json.loads((out/'protocol.json').read_text()) and protocol['version']==2 and protocol['request_token_ceiling']==256
assert source_fingerprint(archive)==protocol['source_fingerprint']==freeze['fingerprint']
assert hashlib.sha256(Path(freeze['zip']).read_bytes()).hexdigest()==freeze['zip_sha256']
with zipfile.ZipFile(freeze['zip']) as bundle:
    assert bundle.testzip() is None and len(bundle.namelist())==149
    assert all(bundle.read(name)==(archive/name).read_bytes() for name in bundle.namelist())
reference=json.loads((work.parent/'outputs/t3-public-host-20261005-format-enforced/protocol.json').read_text())
assert protocol['host']==reference['host'] and protocol['reference_decode']==reference['decode'] and protocol['decode']==dict(reference['decode'],max_new_tokens=256) and protocol['format_backend']==reference['format_backend']
reference_audit=work.parent/'outputs/neural-cbr-t3-format-host-verification.json'
assert hashlib.sha256(reference_audit.read_bytes()).hexdigest()==protocol['reference_verification_sha256']
for file in protocol['host']['files']:
    path=work/'t3-models/qwen3-0.6b'/file['name']
    assert path.stat().st_size==file['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256']
samples_path=work.parent/'outputs/t3-public-data-audit-20261005-preserved-gaps/verification.json'
samples=json.loads(samples_path.read_text())['samples']['squad_train_v11.json']
assert samples['fit_ids']==protocol['fit_ids'] and samples['tune_ids']==protocol['tune_ids']
assert hashlib.sha256(samples_path.read_bytes()).hexdigest()==protocol['sample_sha256']
tasks,_=load_squad(work/'t3-public-data-20261005/squad_train_v11.json',expected_sha256=protocol['data_sha256'])
by_id={t.question_id:t for t in tasks};chosen=[by_id[qid] for qid in protocol['fit_ids']+protocol['tune_ids']]
assert len(chosen)==192 and len({t.question_id for t in chosen})==192
assert {by_id[qid].group for qid in protocol['fit_ids']}.isdisjoint({by_id[qid].group for qid in protocol['tune_ids']})
queries,binding=load_need_queries(out/'need_queries.json',chosen,data_sha256=protocol['data_sha256'],sample_sha256=protocol['sample_sha256'])
records=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
calls=[json.loads(line) for line in (out/'host_calls.jsonl').read_text().splitlines()]
summary=json.loads((out/'summary.json').read_text())
assert records==artifact['records'] and len(records)==len(calls)==summary['calls']==192 and summary['errors']==0
tokenizer=AutoTokenizer.from_pretrained(work/'t3-models/qwen3-0.6b',local_files_only=True,trust_remote_code=False)
for task,record,stored in zip(chosen,records,calls):
    assert record['question_id']==stored['question_id']==task.question_id and record['question']==task.question
    assert record['call']==stored['call'];call=record['call']
    expected=request_prompt(task.question)
    public=tokenizer.apply_chat_template([dict(role='user',content=expected)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert call['source_prompt']==expected and call['prompt']==public and call['stage']=='request'
    assert len(tokenizer(public)['input_ids'])==call['input_tokens'] and 1<=call['output_tokens']<=256
    decision=validate_decision(json.loads(call['output'].strip()),'request')
    assert decision==record['request'] and queries[task.question_id]==decision['need']
assert sum(c['call']['input_tokens'] for c in calls)==summary['input_tokens']
assert sum(c['call']['output_tokens'] for c in calls)==summary['output_tokens']<=protocol['maximum_generated_tokens']
report=dict(source=freeze,queries=192,fit_queries=128,tune_queries=64,public_prompt_token_checks=192,all_recorded_checks_passed=True,
    need_binding=binding,input_artifacts={name:dict(path=str((out/name).resolve()),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in ('protocol.json','need_queries.json','records.jsonl','host_calls.jsonl','summary.json')},
    scope='actual emitted192 train-only requests; exact tokenizer/source prompts,identity/data/host/decoder/output/journal coverage; no fresh LLM generation,semantic usefulness or global promotion claim')
(work.parent/'outputs/neural-cbr-t3-train-need-query-256-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in {'source','need_binding','input_artifacts'}}))
