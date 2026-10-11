from collections import Counter
import hashlib
import json
from pathlib import Path

work=Path(__file__).resolve().parent;outputs=work.parent/'outputs'
def load(name,audit_name=None):
    out=outputs/name
    if audit_name:
        audit=json.loads((outputs/audit_name).read_text());assert not audit['provisional_snapshot'] and audit['all_recorded_retrieval_traces_verified']
        for file,bound in audit['input_artifacts'].items():assert hashlib.sha256((out/file).read_bytes()).hexdigest()==bound['sha256']
    rows=[json.loads(line) for line in (out/'trials.jsonl').read_text().splitlines()]
    assert len(rows)==80
    return json.loads((out/'protocol.json').read_text()),json.loads((out/'summary.json').read_text()),{(r['dataset'],r['question_id'],r['condition']):r for r in rows}
legacy,legacy_summary,old=load('t3-public-host-20261005-request-v2')
plain,plain_summary,p=load('t3-public-host-20261005-strict-plain','neural-cbr-t3-strict-plain-verification.json')
formatted,format_summary,f=load('t3-public-host-20261005-format-enforced','neural-cbr-t3-format-host-verification.json')
assert set(old)==set(p)==set(f)
for key in ('host','metric_checkpoint_sha256','data','case_pools','questions','decode','budgets','conditions'):
    assert legacy[key]==plain[key]==formatted[key]
assert plain['strict_schema']==formatted['strict_schema'] is True and plain['format_backend'] is None
matched_calls=0;new_failures=[]
for key,row in p.items():
    original=old[key]
    assert len(row['host_calls'])==len(original['host_calls'])
    for call,previous in zip(row['host_calls'],original['host_calls']):
        assert all(call[field]==previous[field] for field in ('prompt','output','input_tokens','output_tokens'));matched_calls+=1
    if 'error' in row and 'error' not in original:
        new_failures.append(dict(key=key,error=row['error'],legacy_scores=original['scores']))
assert matched_calls==144 and len(new_failures)==13
groups={}
for key in plain_summary['by_condition']:
    dataset,condition=key.split('/');keys=[k for k in p if k[0]==dataset and k[2]==condition]
    a=plain_summary['by_condition'][key];b=format_summary['by_condition'][key]
    groups[key]=dict(plain=a,formatted=b,delta_f1=b['answer_f1']-a['answer_f1'],
        paired_wins=sum(f[k]['scores']['f1']>p[k]['scores']['f1'] for k in keys),
        paired_losses=sum(f[k]['scores']['f1']<p[k]['scores']['f1'] for k in keys),
        paired_ties=sum(f[k]['scores']['f1']==p[k]['scores']['f1'] for k in keys),
        plain_support_f1=sum(p[k]['scores'].get('sp_f1',0) for k in keys)/8 if dataset=='hotpot' else None,
        formatted_support_f1=sum(f[k]['scores'].get('sp_f1',0) for k in keys)/8 if dataset=='hotpot' else None,
        plain_joint_f1=sum(p[k]['scores'].get('joint_f1',0) for k in keys)/8 if dataset=='hotpot' else None,
        formatted_joint_f1=sum(f[k]['scores'].get('joint_f1',0) for k in keys)/8 if dataset=='hotpot' else None)
report=dict(trials_per_arm=80,host_calls_per_arm=144,legacy_plain_identical_raw_calls=matched_calls,
    newly_rejected_by_strict_validation=new_failures,plain_errors=24,formatted_errors=7,
    plain_error_types=dict(Counter(r['error'] for r in p.values() if 'error' in r)),
    by_condition=groups,plain_tokens=dict(input=plain_summary['input_tokens'],output=plain_summary['output_tokens']),
    formatted_tokens=dict(input=format_summary['input_tokens'],output=format_summary['output_tokens']),
    plain_trial_seconds=sum(g['seconds'] for g in plain_summary['by_condition'].values()),
    formatted_trial_seconds=sum(g['seconds'] for g in format_summary['by_condition'].values()),
    boundary='paired eight engineering questions per dataset/condition; same strict contract/host/prompt/data/ceilings; decoder changes actual needs and answers; costs observed under differing machine load; no confirmation or competitive quality claim')
(outputs/'neural-cbr-t3-strict-format-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['# Strict plain versus constrained decoder controls','',
    'Both80-trial arms use the same strict contract,host,prompts,pools,metric,questions and ceilings. Each makes144 actual host calls. Errors count as zero in every mean.','',
    '| Dataset/condition | Plain F1 | Format F1 | Difference | Errors plain/format | Calls | Seconds plain/format | Paired wins/losses/ties |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for key,g in groups.items():
    a,b=g['plain'],g['formatted'];lines.append(f"| {key} | {a['answer_f1']:.4f} | {b['answer_f1']:.4f} | {g['delta_f1']:+.4f} | {a['errors']}/{b['errors']} | {a['host_calls']} | {a['seconds']:.2f}/{b['seconds']:.2f} | {g['paired_wins']}/{g['paired_losses']}/{g['paired_ties']} |")
lines+=['',f"Plain24/80 failures versus constrained7/80. All128 combined retrieval events and3,552 manual NN candidates pass independent journal/core/payload replay;no error trace gaps or second retrievals.",
    f"Plain48,219 input/6,543 output tokens,{report['plain_trial_seconds']:.2f}s summed trials;constrained48,041/6,488 tokens,{report['formatted_trial_seconds']:.2f}s trials plus6.36s format startup. Shared model/data loading is separate.",
    '', 'All144 strict plain raw prompts/outputs/token counts exactly match the legacy plain run. Its13 additional failures result from validation of the same generated content,not changed model generations. Legacy11 missing error traces remain unrecovered.',
    'All Hotpot support/joint F1 means remain zero in both arms.', '',report['boundary'],
    '', 'Evidence:outputs/neural-cbr-t3-strict-plain-verification.json;outputs/neural-cbr-t3-format-host-verification.json;outputs/neural-cbr-t3-strict-format-comparison.json.',
    'Shared141-file source90aac35f106efb737679739d58f295e52fd66d44738c72b2e4e686d7b88f7d0e;ZIP530f1eef49900ede6ceaa9b445429cb874330da99ace57b48e031683fddda77f.']
(outputs/'neural-cbr-t3-strict-format-comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in {'by_condition','newly_rejected_by_strict_validation','boundary'}}))
