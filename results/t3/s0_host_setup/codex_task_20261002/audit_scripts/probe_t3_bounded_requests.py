import hashlib
import json
import os
from pathlib import Path
import sys
import time
work=Path(__file__).resolve().parent
native=Path('C:/Users/Administrator/NN-KNN/nnknn-work/.venv/Library/bin')
os.environ['PATH']=str(native)+os.pathsep+os.environ.get('PATH','')
handle=os.add_dll_directory(str(native)) if sys.platform=='win32' else None
freeze=json.loads((work/'t3_bounded_request_256_freeze.json').read_text())
sys.path.insert(0,freeze['archive']);sys.path.insert(0,str(work/'t3-format-libs'))
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from model.t1.artifacts import source_fingerprint
from model.t3.host_schema import request_prompt,decoder_schema,validate_decision
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn
assert source_fingerprint(Path(freeze['archive']))==freeze['fingerprint']
source_path=work.parent/'outputs/t3-train-need-queries-256-20261005/need_queries.json'
source=json.loads(source_path.read_text());host=source['protocol']['host']
chosen=[row for row in source['records'] if row['error'] is not None]
assert len(chosen)==2
model_dir=work/'t3-models/qwen3-0.6b'
for row in host['files']:
    file=model_dir/row['name']
    assert file.stat().st_size==row['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
out=work.parent/'outputs/t3-bounded-request-diagnostic-20261006';out.mkdir(exist_ok=False)
torch.manual_seed(0);torch.set_num_threads(2)
tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
token_data=build_token_enforcer_tokenizer_data(tokenizer)
model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,dtype=torch.float16,attn_implementation='eager').to('xpu').eval().requires_grad_(False)
records=[]
for previous in chosen:
    prompt=request_prompt(previous['question'])
    public=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert public==previous['call']['prompt']
    inputs=tokenizer(public,return_tensors='pt').to('xpu')
    fn=build_transformers_prefix_allowed_tokens_fn(token_data,JsonSchemaParser(decoder_schema('request',request_profile='bounded')))
    start=time.perf_counter()
    with torch.no_grad():generated=model.generate(**inputs,**source['protocol']['decode'],prefix_allowed_tokens_fn=fn,pad_token_id=tokenizer.eos_token_id)
    torch.xpu.synchronize()
    text=tokenizer.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    record=dict(question_id=previous['question_id'],question=previous['question'],source_prompt=prompt,prompt=public,output=text,
        input_tokens=inputs['input_ids'].shape[1],output_tokens=generated.shape[1]-inputs['input_ids'].shape[1],seconds=time.perf_counter()-start,error=None,request=None)
    try:record['request']=validate_decision(json.loads(text.strip()),'request',request_profile='bounded')
    except (ValueError,KeyError,TypeError) as exc:record['error']=str(exc)
    with (out/'calls.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(record,ensure_ascii=False)+'\n')
    records.append(record);print(record['question_id'],record['output_tokens'],record['error'],flush=True)
report=dict(source=freeze,host=host,decode=source['protocol']['decode'],request_profile='bounded',
    schema=decoder_schema('request',request_profile='bounded'),records=records,
    failed_source=dict(path=str(source_path),sha256=hashlib.sha256(source_path.read_bytes()).hexdigest()),
    all_requests_complete=all(row['error'] is None for row in records),
    scope='two actual postfailure diagnostic requests; no192 coverage,selection-free outcome benefit or fitting claim')
(out/'summary.json').write_text(json.dumps(report,indent=2))
assert report['all_requests_complete']
