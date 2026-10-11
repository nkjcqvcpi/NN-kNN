import hashlib
import json
from pathlib import Path
work=Path(__file__).resolve().parent;root=work.parent/'outputs'
packages={};costs={}
for arm in ('question-control','actual'):
    out=root/f't3-public-host-need-{arm}-bounded-20261006'
    audit=json.loads((root/f'neural-cbr-t3-public-need-{arm}-bounded-verification.json').read_text())
    assert audit['all_recorded_checks_passed'] and audit['all_recorded_retrieval_traces_verified'] and audit['counts']['trials']==32
    for name,bound in audit['input_artifacts'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==bound['sha256']
    protocol=json.loads((out/'protocol.json').read_text());summary=json.loads((out/'summary.json').read_text())
    rows=[json.loads(line) for line in (out/'trials.jsonl').read_text().splitlines()]
    assert len(rows)==32 and summary['host_calls']==64
    packages[arm]=dict(protocol=protocol,summary=summary,rows=rows,audit=audit)
    costs[arm]=dict(host_calls=64,input_tokens=summary['input_tokens'],output_tokens=summary['output_tokens'],
        summed_trial_seconds=sum(row['seconds'] for row in rows),format_initialization_seconds=protocol['format_initialization_seconds'])
a,b=[packages[arm]['protocol'] for arm in ('question-control','actual')]
for key in ('source_fingerprint','host','data','conditions','decode','request_decode','request_profile','budgets','questions','case_pools','format_backend'):
    assert a[key]==b[key],key
assert a['metric_selection']['seed']==b['metric_selection']['seed']==8
assert a['metric_selection']['selection']==b['metric_selection']['selection']=='tune_recall'
assert a['metric_selection']['query_source']=='original public question' and b['metric_selection']['query_source']=='actual frozen host request'
before={(r['dataset'],r['question_id'],r['condition']):r for r in packages['question-control']['rows']}
after={(r['dataset'],r['question_id'],r['condition']):r for r in packages['actual']['rows']}
assert before.keys()==after.keys()
groups=[];changed_sets=changed_orders=changed_answers=0
for dataset in ('squad','hotpot'):
    for condition in ('learned','iterative'):
        keys=[key for key in before if key[0]==dataset and key[2]==condition];assert len(keys)==8
        means={};improved=harmed=tied=0
        for arm,data in [('question-control',before),('actual',after)]:
            means[arm]=dict(answer_f1=sum(data[key]['scores']['f1'] for key in keys)/8,
                errors=sum('error' in data[key] for key in keys),
                supporting_f1=sum(data[key]['scores'].get('sp_f1',0.) for key in keys)/8,
                joint_f1=sum(data[key]['scores'].get('joint_f1',0.) for key in keys)/8)
        for key in keys:
            x,y=before[key],after[key]
            assert x['host_calls'][0]['prompt']==y['host_calls'][0]['prompt'] and x['host_calls'][0]['output']==y['host_calls'][0]['output']
            dx=y['scores']['f1']-x['scores']['f1'];improved+=dx>1e-12;harmed+=dx< -1e-12;tied+=abs(dx)<=1e-12
            xe=x['partial_events'] if 'error' in x else x['result']['events'];ye=y['partial_events'] if 'error' in y else y['result']['events']
            assert len(xe)==len(ye)==1
            xi=xe[0]['audit']['selected_ids'];yi=ye[0]['audit']['selected_ids']
            changed_sets+=set(xi)!=set(yi);changed_orders+=xi!=yi
            changed_answers+=x['host_calls'][-1]['output']!=y['host_calls'][-1]['output']
        groups.append(dict(dataset=dataset,condition=condition,means=means,improved=improved,harmed=harmed,tied=tied))
report=dict(groups=groups,costs=costs,actual_trials=64,actual_calls=128,changed_sets=changed_sets,changed_orders=changed_orders,changed_final_outputs=changed_answers,
    identical_initial_requests=32,manual_candidates=sum(p['audit']['counts']['manual_candidates'] for p in packages.values()),
    all_recorded_checks_passed=True,scope='same16 exploratory questions/host/bounded request treatment and seed8; fitted metric input-distribution comparison with downstream observed failures retained; no independent confirmation or competitive gate')
(root/'neural-cbr-t3-actual-need-public-results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
