from pathlib import Path

work=Path(__file__).resolve().parent
source=(work/'audit_t3_public_host.py').read_text()
source=source.replace("freeze=json.loads((work/'t3_public_host_request_v2_freeze.json').read_text())", "import argparse\nap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True);ap.add_argument('--output',required=True);ap.add_argument('--report',required=True);ap.add_argument('--expected-trials',type=int,default=80);args=ap.parse_args()\nfreeze=json.loads(Path(args.freeze).read_text())")
source=source.replace("out=work.parent/'outputs/t3-public-host-20261005-request-v2'", "out=Path(args.output)")
source=source.replace('len(bundle.namelist())==139','len(bundle.namelist())==freeze[\'entries\']')
source=source.replace('records=[];event_ids=[]','''records=[];event_ids=[];expected_journal=[]
from model.t3.host_schema import validate_decision
assert protocol['strict_schema'] is True
journal=[json.loads(line) for line in (out/'retrieval_events.jsonl').read_text().splitlines()]
assert len(journal)>0''')
start=source.index("    if 'error' in row:")
end=source.index('    delivered=[]',start)
source=source[:start]+'''    failed='error' in row
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
        assert call['output_tokens']<=128
        if failed and index==len(calls)-1:continue
        raw=call['output'].strip()
        if raw.startswith('```json') and raw.endswith('```'):raw=raw[7:-3].strip()
        validate_decision(json.loads(raw),call['stage'])
    expected_journal.extend(dict(dataset=dataset,question_id=qid,condition=condition,event=e) for e in events)
'''+source[end:]
source=source.replace("    assert delivered==result.get('evidence',[]) and len(delivered)<=2 and len(events)<=2", "    assert len(delivered)<=2 and len(events)<=2\n    if not failed:assert delivered==result.get('evidence',[])")
start=source.index('    visible={}')
end=source.index('assert len(set((r[',start)
source=source[:start]+'''    if not failed:
        visible={}
        for case in delivered:
            obj=json.loads(case['content'])
            if 'sentences' in obj:visible[obj['title']]=len(obj['sentences'])
        observed=all(isinstance(p,(list,tuple)) and len(p)==2 and type(p[1]) is int and p[0] in visible and 0<=p[1]<visible[p[0]] for p in row['supporting_facts'])
        assert observed==row['citations_visible']
    records.append(dict(dataset=dataset,question_id=qid,condition=condition,scores=score,rounds=len(events),failed=failed,recorded_checks_passed=True))
assert journal==expected_journal
'''+source[end:]
source=source.replace("assert all(len(r['host_calls'])==(1 if r['condition']=='none' else 2) for r in rows)","assert all(len(r['host_calls'])<=3 for r in rows)")
source=source.replace('all_retrieval_traces_verified=False','all_recorded_retrieval_traces_verified=True')
source=source.replace("scope='fresh actual core/hash/BM25/manual-feature/event/score/payload replay of 53 recorded retrieval events; 11 post-retrieval error traces missing and not reconstructed; no fresh LLM generations,causal utility or confirmation claim'", "scope='fresh actual core/hash/BM25/manual-feature/event/score/payload replay including failed-trial partial events; exact immediate journal comparison; no fresh LLM generations or causal utility claim'")
source=source.replace("(work.parent/'outputs/neural-cbr-t3-public-host-verification.json').write_text", "Path(args.report).write_text")
source=source.replace('successful_retrieval_events','recorded_retrieval_events')
source=source.replace('==summary[\'trials\']==80','==summary[\'trials\']==args.expected_trials')
source=source.replace('counts=dict(trials=80,','counts=dict(trials=args.expected_trials,')
source=source.replace("for r in rows))==80","for r in rows))==args.expected_trials")
source=source.replace('report=dict(source=freeze,','report=dict(provisional_snapshot=summary.get(\'provisional_snapshot\',False),source=freeze,')
source=source.replace('report=dict(provisional_snapshot=',"artifact_bindings={name:dict(path=str((out/name).resolve()),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in ('protocol.json','trials.jsonl','retrieval_events.jsonl','summary.json')}\nreport=dict(input_artifacts=artifact_bindings,provisional_snapshot=")
(work/'audit_t3_public_journal.py').write_text(source,encoding='utf-8')
compile(source,'audit_t3_public_journal.py','exec')
print('Journal audit prepared; requires terminal80-trial summary.')
