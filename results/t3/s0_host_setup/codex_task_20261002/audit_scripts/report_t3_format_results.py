import hashlib
import json
import re
from pathlib import Path

work=Path(__file__).resolve().parent
out=work.parent/'outputs/t3-public-host-20261005-format-enforced'
audit=json.loads((work.parent/'outputs/neural-cbr-t3-format-host-verification.json').read_text())
assert not audit['provisional_snapshot'] and audit['all_recorded_retrieval_traces_verified']
for name,record in audit['input_artifacts'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==record['sha256']
protocol=json.loads((out/'protocol.json').read_text());summary=json.loads((out/'summary.json').read_text())
rows=[json.loads(line) for line in (out/'trials.jsonl').read_text().splitlines()]
questions={d:{q['question_id']:q['question'] for q in qs} for d,qs in protocol['questions'].items()}
errors=[dict(dataset=r['dataset'],question_id=r['question_id'],condition=r['condition'],error=r['error'],
    last_output_tokens=r['host_calls'][-1]['output_tokens']) for r in rows if 'error' in r]
groups={}
for key,value in summary['by_condition'].items():
    dataset,condition=key.split('/');selected=[r for r in rows if (r['dataset'],r['condition'])==(dataset,condition)]
    groups[key]=dict(value,input_tokens=sum(c['input_tokens'] for r in selected for c in r['host_calls']),
        output_tokens=sum(c['output_tokens'] for r in selected for c in r['host_calls']),
        capped_calls=sum(c['output_tokens']==128 for r in selected for c in r['host_calls']),
        support_f1=sum(r['scores'].get('sp_f1',0) for r in selected)/len(selected) if dataset=='hotpot' else None,
        joint_f1=sum(r['scores'].get('joint_f1',0) for r in selected)/len(selected) if dataset=='hotpot' else None)
events=[(r,e) for r in rows for e in r.get('result',{}).get('events',r.get('partial_events',[]))]
copied=sum(e['audit']['request']['need']==questions[r['dataset']][r['question_id']] for r,e in events)
lexical_copied=sum(re.findall(r'[a-z0-9]+',e['audit']['request']['need'].lower())==re.findall(r'[a-z0-9]+',questions[r['dataset']][r['question_id']].lower()) for r,e in events)
report=dict(trials=80,counts=audit['counts'],by_condition=groups,input_tokens=summary['input_tokens'],output_tokens=summary['output_tokens'],
    generation_trial_seconds=sum(v['seconds'] for v in groups.values()),format_initialization_seconds=protocol['format_initialization_seconds'],
    errors=errors,exact_question_copy_requests=copied,identical_lexical_token_requests=lexical_copied,actual_requests=len(events),
    limits='descriptive engineering comparison; strict validation and decoder both differ from legacy plain; no decoder causal attribution before strict plain control; no QA or multi-hop success claim')
(work.parent/'outputs/neural-cbr-t3-format-host-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['# Complete constrained public host results','',
    f"80 trials;144 calls;{summary['input_tokens']:,} input/{summary['output_tokens']:,} output tokens;{report['generation_trial_seconds']:.2f}s summed trial time;{protocol['format_initialization_seconds']:.2f}s separate format startup.",'',
    '| Dataset/condition | EM | Answer F1 | Errors/8 | Calls | Seconds | Input/output tokens | Capped calls |','|---|---:|---:|---:|---:|---:|---:|---:|']
for key,g in groups.items():lines.append(f"| {key} | {g['answer_em']:.4f} | {g['answer_f1']:.4f} | {g['errors']} | {g['host_calls']} | {g['seconds']:.2f} | {g['input_tokens']}/{g['output_tokens']} | {g['capped_calls']} |")
lines+=['',f"All {len(events)} actual retrieval events,including seven failed-trial traces,pass full journal/core/payload verification;1,776 manual NN candidates checked. No second retrieval occurs.",
    f"Explicit model needs exactly copy the public question in {copied}/{len(events)} requests;the controller does not replace a differing need.",
    f"The declared ASCII/lowercase lexical token sequence equals the public question in {lexical_copied}/{len(events)} requests;literal punctuation differences do not change this retrieval geometry.",
    'All Hotpot support/joint F1 means remain zero. Error scores stay zero in the denominator.',
    '',report['limits'],'',
    'Source141 files:90aac35f106efb737679739d58f295e52fd66d44738c72b2e4e686d7b88f7d0e;ZIP530f1eef49900ede6ceaa9b445429cb874330da99ace57b48e031683fddda77f.',
    'Evidence: outputs/neural-cbr-t3-format-host-verification.json;outputs/neural-cbr-t3-format-host-results.json;outputs/t3-public-host-20261005-format-enforced/.']
(work.parent/'outputs/neural-cbr-t3-format-host-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in {'by_condition','limits'}}))
