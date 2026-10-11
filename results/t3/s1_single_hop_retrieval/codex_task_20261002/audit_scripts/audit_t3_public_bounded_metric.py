import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch
work=Path(__file__).resolve().parent
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True);ap.add_argument('--output',required=True);ap.add_argument('--report',required=True);ap.add_argument('--expected-trials',type=int,default=80);ap.add_argument('--metric-checkpoint',type=Path,required=True);args=ap.parse_args()
freeze=json.loads(Path(args.freeze).read_text())
archive=Path(freeze['archive']);sys.path.insert(0,str(archive))
from model.t1.artifacts import source_fingerprint
from model.t3.benchmarks import load_squad,load_hotpot,answer_metrics,hotpot_metrics
from model.t3.lexical import core_bank,HashQuery,bm25_rank,hash_vector
from model.t3.retrieval import Case,Retriever,Access,Request
out=Path(args.output)
protocol=json.loads((out/'protocol.json').read_text())
assert protocol['source_fingerprint']==freeze['fingerprint']==source_fingerprint(archive)
assert hashlib.sha256(Path(freeze['zip']).read_bytes()).hexdigest()==freeze['zip_sha256']
with zipfile.ZipFile(freeze['zip']) as bundle:
    assert bundle.testzip() is None and len(bundle.namelist())==freeze['entries']
    assert all(bundle.read(name)==(archive/name).read_bytes() for name in bundle.namelist())
metric_path=args.metric_checkpoint
from model.t3.metric_provenance import metric_provenance
assert protocol['metric_selection']==metric_provenance(metric_path)
assert hashlib.sha256(metric_path.read_bytes()).hexdigest()==protocol['metric_checkpoint_sha256']
metric=torch.load(metric_path,weights_only=True)['selected_metric']['feature_weights']
files={r['name']:r for r in protocol['data']['files']}
golds={}
for dataset,name,loader in [('squad','squad_dev_v11.json',load_squad),('hotpot','hotpot_dev_distractor_converted.json',load_hotpot)]:
    tasks,gold=loader(work/'t3-public-data-20261005'/name,expected_sha256=files[name]['sha256'],**(dict(allow_unaligned_support=True) if dataset=='hotpot' else {}))
    golds[dataset]=gold
    actual={t.question_id:t for t in tasks}
    for public in protocol['questions'][dataset]:assert actual[public['question_id']].host_task()==public
summary=json.loads((out/'summary.json').read_text())
rows=[json.loads(line) for line in (out/'trials.jsonl').read_text().splitlines()]
assert len(rows)==summary['trials']==args.expected_trials
counts=dict(trials=args.expected_trials,recorded_retrieval_events=0,nn_events=0,bm25_events=0,manual_candidates=0,
            host_calls=0,schema_failures=0,second_retrieval_trials=0,error_trace_gaps=0)
records=[];event_ids=[];expected_journal=[]
from model.t3.host_schema import validate_decision
assert protocol['strict_schema'] is True
journal=[json.loads(line) for line in (out/'retrieval_events.jsonl').read_text().splitlines()]
assert len(journal)>0
for row in rows:
    dataset,qid,condition=row['dataset'],row['question_id'],row['condition']
    assert dataset in protocol['questions'] and condition in protocol['conditions']
    bank=[Case(**c) for c in protocol['case_pools'][dataset][qid]]
    calls=row['host_calls'];counts['host_calls']+=len(calls)
    assert len(calls)<=3 and sum(c['output_tokens'] for c in calls)<=protocol['budgets']['total_generated_token_ceiling']
    failed='error' in row
    if failed:
        assert row['scores']['em']==row['scores']['f1']==0
        counts['schema_failures']+=1
        events=row['partial_events'];score=row['scores']
    else:
        result=row['result'];events=result['events']
        score=hotpot_metrics(result.get('answer') or '',row['supporting_facts'],golds[dataset][qid]) if dataset=='hotpot' else answer_metrics(result.get('answer') or '',golds[dataset][qid])
        assert score==row['scores']
    for index,call in enumerate(calls):
        assert call['stage'] in {'request','answer','continuation'}
        assert call['output_tokens']<=(protocol['request_decode']['max_new_tokens'] if call['stage']=='request' else protocol['decode']['max_new_tokens'])
        if failed and index==len(calls)-1:continue
        raw=call['output'].strip()
        if raw.startswith('```json') and raw.endswith('```'):raw=raw[7:-3].strip()
        validate_decision(json.loads(raw),call['stage'],request_profile=protocol['request_profile'])
    expected_journal.extend(dict(dataset=dataset,question_id=qid,condition=condition,event=e) for e in events)
    delivered=[]
    if condition not in {'none','bm25'}:
        model=core_bank(bank)
        if condition in {'learned','iterative'}:model.glocal_weightor.feature_weights.data.copy_(metric)
        model.eval().requires_grad_(False)
        retriever=Retriever(model,bank,HashQuery(),model_version='fixed' if condition=='fixed' else f"need-s{protocol['metric_selection']['seed']}-{protocol['metric_checkpoint_sha256'][:12]}",encoder_version='hash256-v1')
    for event in events:
        audit=event['audit'];event_ids.append(event['event_id'])
        assert audit['event_id']==event['event_id']
        request=Request(**audit['request'])
        assert tuple(c['case_id'] for c in delivered)==request.already_retrieved_ids
        assert request.max_cases==(1 if condition=='iterative' else 2)
        if condition=='bm25':
            eligible=[c for c in bank if c.case_id not in request.already_retrieved_ids and c.artifact_type in request.requested_types]
            ranking=bm25_rank(request.need,eligible)
            assert ranking==audit['candidates']
            assert [r['case_id'] for r in ranking[:request.max_cases]]==audit['selected_ids']
            counts['bm25_events']+=1
        else:
            fresh=retriever.retrieve(request,Access('public-benchmark','public-benchmark'),now=audit['timestamp'])
            assert fresh['evidence']==event['evidence']
            for key in ('selected_ids','model_state_sha256','encoder_state_sha256','encoder_verification','bank_version'):
                assert fresh['audit'][key]==audit[key]
            q=hash_vector(request.need)
            contrib=(model.cases-q).square()*model.glocal_weightor.feature_weights[0]
            distances=contrib.sum(-1).sqrt()
            for col,candidate in enumerate(audit['candidates']):
                slot=next(i for i,c in enumerate(bank) if c.case_id==candidate['case_id'])
                torch.testing.assert_close(torch.tensor(candidate['distance']),distances[slot],rtol=1e-6,atol=1e-7)
                torch.testing.assert_close(torch.tensor(candidate['feature_distance_contributions']),contrib[slot],rtol=0,atol=0)
                torch.testing.assert_close(torch.tensor(candidate['weight']),torch.tensor(fresh['audit']['candidates'][col]['weight']),rtol=0,atol=0)
                counts['manual_candidates']+=1
            counts['nn_events']+=1
        if event.get('delivered_to_host'):delivered.extend(event['evidence'])
        counts['recorded_retrieval_events']+=1
    assert len(delivered)<=2 and len(events)<=2
    if not failed:assert delivered==result.get('evidence',[])
    counts['second_retrieval_trials']+=int(len(events)==2)
    for call in calls[1:]:
        # Every prompt carries precisely a delivered prefix; never all pool
        # paragraphs or hidden annotations. Match real payload bytes.
        assert any(json.dumps(delivered[:n],ensure_ascii=False) in call['prompt'] for n in range(1,len(delivered)+1))
    if not failed:
        visible={}
        for case in delivered:
            obj=json.loads(case['content'])
            if 'sentences' in obj:visible[obj['title']]=len(obj['sentences'])
        observed=all(isinstance(p,(list,tuple)) and len(p)==2 and type(p[1]) is int and p[0] in visible and 0<=p[1]<visible[p[0]] for p in row['supporting_facts'])
        assert observed==row['citations_visible']
    records.append(dict(dataset=dataset,question_id=qid,condition=condition,scores=score,rounds=len(events),failed=failed,recorded_checks_passed=True))
assert journal==expected_journal
assert len(set((r['dataset'],r['question_id'],r['condition']) for r in rows))==args.expected_trials
assert len(set(event_ids))==len(event_ids)
assert counts['host_calls']==summary['host_calls']
assert sum(c['input_tokens'] for row in rows for c in row['host_calls'])==summary['input_tokens']
assert sum(c['output_tokens'] for row in rows for c in row['host_calls'])==summary['output_tokens']
assert all(len(r['host_calls'])<=3 for r in rows)
artifact_bindings={name:dict(path=str((out/name).resolve()),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in ('protocol.json','trials.jsonl','retrieval_events.jsonl','summary.json')}
report=dict(input_artifacts=artifact_bindings,provisional_snapshot=summary.get('provisional_snapshot',False),source=freeze,counts=counts,records=records,all_recorded_checks_passed=True,all_recorded_retrieval_traces_verified=True,
    scope='fresh actual core/hash/BM25/manual-feature/event/score/payload replay including failed-trial partial events; exact immediate journal comparison; no fresh LLM generations or causal utility claim')
Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(counts))
