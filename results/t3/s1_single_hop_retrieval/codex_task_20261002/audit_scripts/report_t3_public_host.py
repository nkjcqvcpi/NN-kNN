import json
from pathlib import Path
from collections import Counter

work=Path(__file__).resolve().parent
out=work.parent/'outputs/t3-public-host-20261005-request-v2'
rows=[json.loads(line) for line in (out/'trials.jsonl').read_text().splitlines()]
summary=json.loads((out/'summary.json').read_text())
audit=json.loads((work.parent/'outputs/neural-cbr-t3-public-host-verification.json').read_text())
assert len(rows)==80 and audit['counts']['error_trace_gaps']==11
groups={}
for key,s in summary['by_condition'].items():
    dataset,condition=key.split('/')
    group=[r for r in rows if (r['dataset'],r['condition'])==(dataset,condition)]
    groups[key]=dict(s,input_tokens=sum(c['input_tokens'] for r in group for c in r['host_calls']),
        output_tokens=sum(c['output_tokens'] for r in group for c in r['host_calls']),
        capped_calls=sum(c['output_tokens']==128 for r in group for c in r['host_calls']),
        recorded_retrievals=sum(len(r.get('result',{}).get('events',[])) for r in group),
        supporting_f1=sum(r['scores'].get('sp_f1',0) for r in group)/len(group) if dataset=='hotpot' else None,
        joint_f1=sum(r['scores'].get('joint_f1',0) for r in group)/len(group) if dataset=='hotpot' else None)
errors=[dict(dataset=r['dataset'],question_id=r['question_id'],condition=r['condition'],error=r['error'],
    last_output_tokens=r['host_calls'][-1]['output_tokens'],last_output=r['host_calls'][-1]['output']) for r in rows if 'error' in r]
assert len(errors)==11
report=dict(trials=80,host_calls=144,input_tokens=summary['input_tokens'],output_tokens=summary['output_tokens'],
    by_condition=groups,errors=errors,error_messages=dict(Counter(e['error'] for e in errors)),
    audit_boundary=audit['scope'],actual_second_retrievals=0,
    limitations=['eight prospective engineering questions per dataset; no confirmatory inference',
    'same resource ceilings do not equal actual compute: none uses one call,all other trials use two',
    'iterative supplies one case and no host requests a second round; not a demonstrated multi-hop loop',
    'conditional 64-case SQuAD pools and Hotpot distractor contexts,not unrestricted corpus retrieval',
    'old errored retrieval event IDs missing; new journal cannot reconstruct them'])
(work.parent/'outputs/neural-cbr-t3-public-host-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['# Public host engineering results','',
    '80 trials; 144 host calls; 48,219 input tokens; 6,543 output tokens. Errors count as zero in all means.','',
    '| Dataset/condition | EM | Answer F1 | Errors | Calls | Seconds | Input/output tokens | Capped calls |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for key,g in groups.items():
    lines.append(f"| {key} | {g['answer_em']:.4f} | {g['answer_f1']:.4f} | {g['errors']} | {g['host_calls']} | {g['seconds']:.2f} | {g['input_tokens']}/{g['output_tokens']} | {g['capped_calls']} |")
lines+=['','Supporting/joint Hotpot F1: '+str({k:(v['supporting_f1'],v['joint_f1']) for k,v in groups.items() if k.startswith('hotpot/')}),
    '',audit['scope'],'']+['- '+s for s in report['limitations']]
(work.parent/'outputs/neural-cbr-t3-public-host-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:{n:v[n] for n in ('capped_calls','supporting_f1','joint_f1')} for k,v in groups.items()}))
print(json.dumps(report['error_messages']))
