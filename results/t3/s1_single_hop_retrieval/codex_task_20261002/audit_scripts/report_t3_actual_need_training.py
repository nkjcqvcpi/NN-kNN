import hashlib
import json
from pathlib import Path
work=Path(__file__).resolve().parent;root=work.parent/'outputs'
replay=json.loads((root/'neural-cbr-t3-actual-need-bounded-retraining-verification.json').read_text())
common=json.loads((root/'neural-cbr-t3-common-need-bounded-evaluation.json').read_text())
assert replay['all_recorded_checks_passed'] and replay['counts']['actual_retrainings']==6
assert common['all_recorded_checks_passed'] and common['counts']['actual_nn_evaluations']==2688
for package in ('question-control','actual'):
    path=root/f't3-train-need-{package}-bounded-20261006'
    aggregate=json.loads((path/'summary.json').read_text())
    assert len(aggregate['runs'])==3
    for run in aggregate['runs']:assert hashlib.sha256((path/f"s{run['seed']}/metric.pt").read_bytes()).hexdigest()==run['checkpoint_sha256']
records=[row for row in common['records'] if row['group']=='tune']
by_key={(row['query_source'],row['condition']):row for row in records}
paired=[]
for query in ('original_question','actual_need'):
    for seed in (8,9,10):
        a,b=[by_key[(query,f'{name}/s{seed}')] for name in ('question-control','actual')]
        before={row['question_id']:row for row in a['events']};after={row['question_id']:row for row in b['events']}
        assert before.keys()==after.keys() and len(before)==64
        improved=harmed=0;changed=0
        for qid,x in before.items():
            y=after[qid];assert x['need']==y['need'] and x['positive_ids']==y['positive_ids']
            old=bool(set(x['selected_ids'])&set(x['positive_ids']));new=bool(set(y['selected_ids'])&set(y['positive_ids']))
            improved+=int(new and not old);harmed+=int(old and not new)
            changed+=x['selected_ids']!=y['selected_ids']
        paired.append(dict(query_source=query,seed=seed,before_recall2=a['source_recall_at2'],after_recall2=b['source_recall_at2'],improved=improved,harmed=harmed,changed_selected_orders=changed))
table=[{key:row[key] for key in ('query_source','condition','questions','source_recall_at1','source_recall_at2')} for row in records]
report=dict(retraining_counts=replay['counts'],common_evaluation_counts=common['counts'],tune=table,paired=paired,
    all_recorded_checks_passed=True,scope='same-context source identity/selection-data diagnostics; not downstream utility or independent confirmation')
(root/'neural-cbr-t3-actual-need-training-results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
