"""Collect actual frozen-host needs for the prespecified SQuAD train fit/tune IDs.

Same request prompt/checkpoint/decoder as public evaluation; question-only
payload,no answers/passages/dev fitting,no fallback or missing-query filtering.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-run','reference-verification','data-root','samples','model-dir','runtime-library-bin','format-library','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--request-token-ceiling',type=int,choices=(128,256,512),default=128)
    ap.add_argument('--request-profile',choices=('standard','bounded'),default='standard')
    args=ap.parse_args()
    verification=json.loads(args.reference_verification.read_text())
    if verification.get('provisional_snapshot') or not verification.get('all_recorded_retrieval_traces_verified') or verification['counts']['trials']!=80:
        raise ValueError('complete audited frozen-host reference required')
    for name,bound in verification['input_artifacts'].items():
        if Path(bound['path']).resolve()!=(args.reference_run/name).resolve() or sha(args.reference_run/name)!=bound['sha256']:
            raise ValueError('reference byte/path binding changed')
    reference=json.loads((args.reference_run/'protocol.json').read_text())
    if not reference['strict_schema'] or reference['host']['revision']!='c1899de289a04d12100db370d81485cdf75e47ca' or reference['decode']['max_new_tokens']!=128:
        raise ValueError('verified strict request treatment required')
    native=args.runtime_library_bin.resolve(strict=True)
    os.environ['PATH']=str(native)+os.pathsep+os.environ.get('PATH','')
    native_handle=os.add_dll_directory(str(native)) if sys.platform=='win32' else None
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from model.t1.artifacts import source_fingerprint
    from model.t3.benchmarks import load_squad
    from model.t3.host_schema import request_prompt,request_decode,decision_schema,decoder_schema,validate_decision
    from model.t3.need_queries import load_need_queries
    sys.path.insert(0,str(args.format_library.resolve(strict=True)))
    import importlib.metadata
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn
    if importlib.metadata.version('lm-format-enforcer')!='0.11.3':raise ValueError('verified backend version required')
    backend=hashlib.sha256()
    for path in sorted((args.format_library/'lmformatenforcer').rglob('*.py')):
        backend.update(path.relative_to(args.format_library/'lmformatenforcer').as_posix().encode());backend.update(path.read_bytes())
    if backend.hexdigest()!=reference['format_backend']['backend_source_sha256']:raise ValueError('backend drift')
    host=reference['host'];names=set()
    for record in host['files']:
        name=record['name']
        if name!=Path(name).name or name in names:raise ValueError('invalid host manifest')
        names.add(name);path=args.model_dir/name
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise ValueError('host file drift')
    if names!={p.name for p in args.model_dir.iterdir() if p.is_file()}:raise ValueError('host manifest incomplete')
    source=next(r for r in reference['data']['files'] if r['name']=='squad_train_v11.json')
    tasks,_=load_squad(args.data_root/source['name'],expected_sha256=source['sha256'])
    samples=json.loads(args.samples.read_text())['samples'][source['name']]
    if len(samples['fit_ids'])!=128 or len(samples['tune_ids'])!=64 or set(samples['fit_ids'])&set(samples['tune_ids']):
        raise ValueError('complete prespecified128/64 sample required')
    by_id={t.question_id:t for t in tasks}
    fit=[by_id[qid] for qid in samples['fit_ids']];tune=[by_id[qid] for qid in samples['tune_ids']]
    if {t.group for t in fit}&{t.group for t in tune}:raise ValueError('article leakage')
    decode=request_decode(reference['decode'],args.request_token_ceiling)
    format_backend=dict(reference['format_backend'],
        schemas={stage:decision_schema(stage,request_profile=args.request_profile) for stage in ('request','answer','continuation')},
        decoder_schemas={stage:decoder_schema(stage,request_profile=args.request_profile) for stage in ('request','answer','continuation')})
    protocol=dict(version=1 if args.request_token_ceiling==128 and args.request_profile=='standard' else 2,dataset=source['name'],data_sha256=source['sha256'],sample_sha256=sha(args.samples),
        source_fingerprint=source_fingerprint(Path(__file__).resolve().parents[1]),host=host,format_backend=format_backend,
        reference_format_backend=reference['format_backend'],request_profile=args.request_profile,
        reference_verification_sha256=sha(args.reference_verification),query_source='actual frozen host request',
        public_dev_annotations_used=False,fit_ids=samples['fit_ids'],tune_ids=samples['tune_ids'],
        decode=decode,request_token_ceiling=args.request_token_ceiling,
        request_treatment='uniform request cap; same prompt/host/grammar; no per-question retry',
        reference_decode=reference['decode'],maximum_calls=192,maximum_generated_tokens=192*args.request_token_ceiling,
        trainable_host_parameters=0,missing='failed needs remain unobserved; no fallback/filter/refill; downstream fitter rejects incomplete manifest')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.manual_seed(0);torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
    start=time.perf_counter();token_data=build_token_enforcer_tokenizer_data(tokenizer)
    protocol['format_initialization_seconds']=time.perf_counter()-start
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False,dtype=torch.float16,attn_implementation='eager').to('xpu').eval().requires_grad_(False)
    records=[]
    def append(name,value):
        with (args.output/name).open('a',encoding='utf-8') as file:file.write(json.dumps(value,ensure_ascii=False)+'\n')
    for task in fit+tune:
        prompt=request_prompt(task.question)
        public=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(public,return_tensors='pt').to('xpu');start=time.perf_counter()
        fn=build_transformers_prefix_allowed_tokens_fn(token_data,JsonSchemaParser(decoder_schema('request',request_profile=args.request_profile)))
        with torch.no_grad():output=model.generate(**inputs,**decode,prefix_allowed_tokens_fn=fn,pad_token_id=tokenizer.eos_token_id)
        torch.xpu.synchronize()
        text=tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        call=dict(stage='request',source_prompt=prompt,prompt=public,output=text,input_tokens=inputs['input_ids'].shape[1],
            output_tokens=output.shape[1]-inputs['input_ids'].shape[1],seconds=time.perf_counter()-start)
        append('host_calls.jsonl',dict(question_id=task.question_id,call=call))
        record=dict(question_id=task.question_id,question=task.question,call=call,request=None,error=None)
        try:record['request']=validate_decision(json.loads(text.strip()),'request',request_profile=args.request_profile)
        except (ValueError,KeyError,TypeError) as exc:record['error']=str(exc)
        records.append(record);append('records.jsonl',record)
        print(len(records),task.question_id,'observed' if record['error'] is None else record['error'],flush=True)
    artifact=dict(protocol=protocol,records=records)
    (args.output/'need_queries.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    summary=dict(calls=len(records),valid_requests=sum(r['error'] is None for r in records),errors=sum(r['error'] is not None for r in records),
        input_tokens=sum(r['call']['input_tokens'] for r in records),output_tokens=sum(r['call']['output_tokens'] for r in records),
        generation_seconds=sum(r['call']['seconds'] for r in records))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    load_need_queries(args.output/'need_queries.json',fit+tune,data_sha256=source['sha256'],sample_sha256=sha(args.samples))


if __name__=='__main__':main()
