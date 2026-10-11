import hashlib
import json
import os
from pathlib import Path
import sys
work=Path(__file__).resolve().parent
native=Path('C:/Users/Administrator/NN-KNN/nnknn-work/.venv/Library/bin')
os.environ['PATH']=str(native)+os.pathsep+os.environ.get('PATH','')
handle=os.add_dll_directory(str(native)) if sys.platform=='win32' else None
freeze=json.loads((work/'t3_bounded_request_256_freeze.json').read_text())
sys.path.insert(0,freeze['archive'])
from model.t3.host_schema import request_prompt,decoder_schema,validate_decision
from transformers import AutoTokenizer
out=work.parent/'outputs/t3-bounded-request-diagnostic-20261006'
summary=json.loads((out/'summary.json').read_text())
calls=[json.loads(line) for line in (out/'calls.jsonl').read_text().splitlines()]
assert calls==summary['records'] and len(calls)==2 and summary['all_requests_complete']
assert summary['source']==freeze and summary['request_profile']=='bounded'
assert summary['schema']==decoder_schema('request',request_profile='bounded')
path=Path(summary['failed_source']['path'])
assert hashlib.sha256(path.read_bytes()).hexdigest()==summary['failed_source']['sha256']
old=json.loads(path.read_text())
failures={r['question_id']:r for r in old['records'] if r['error'] is not None}
assert set(failures)=={r['question_id'] for r in calls}
assert summary['host']==old['protocol']['host'] and summary['decode']==old['protocol']['decode']
tokenizer=AutoTokenizer.from_pretrained(work/'t3-models/qwen3-0.6b',local_files_only=True,trust_remote_code=False)
for row in calls:
    previous=failures[row['question_id']]
    assert row['question']==previous['question'] and row['source_prompt']==request_prompt(row['question'])
    public=tokenizer.apply_chat_template([dict(role='user',content=row['source_prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert row['prompt']==public==previous['call']['prompt']
    assert row['input_tokens']==len(tokenizer(public)['input_ids']) and 1<=row['output_tokens']<=256
    assert row['error'] is None and validate_decision(json.loads(row['output']),'request',request_profile='bounded')==row['request']
    assert 500<=len(row['request']['need'])<=512
    assert previous['call']['output'].split('"need": "',1)[1].startswith(row['request']['need'])
report=dict(requests=2,all_recorded_checks_passed=True,input_tokens=sum(r['input_tokens'] for r in calls),
    output_tokens=sum(r['output_tokens'] for r in calls),generation_seconds=sum(r['seconds'] for r in calls),
    exact_initial_repetition_preserved=True,input_artifacts={name:dict(path=str((out/name).resolve()),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in ('calls.jsonl','summary.json')},
    need_lengths=[len(row['request']['need']) for row in calls],
    boundary='two postfailure actual diagnostic calls close original repetition within512 characters; complete schema and exact tokenizer/source bindings, not192 coverage or semantic repair')
(work.parent/'outputs/neural-cbr-t3-bounded-request-probe-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='input_artifacts'}))
