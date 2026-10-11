import hashlib
import json
from pathlib import Path
import sys
import zipfile

work=Path(__file__).resolve().parent
freeze=json.loads((work/'t3_public_set_utility_task_failure_freeze.json').read_text())
archive=Path(freeze['archive']);sys.path.insert(0,str(archive))
from model.t1.artifacts import source_fingerprint
from model.t3.benchmarks import load_squad,load_hotpot,answer_metrics,hotpot_metrics
from model.t3.host_schema import validate_decision
from model.t3.utility import direct_set_audit
out=work.parent/'outputs/t3-public-set-utility-20261005'
protocol=json.loads((out/'protocol.json').read_text())
assert protocol['source_fingerprint']==freeze['fingerprint']==source_fingerprint(archive)
assert hashlib.sha256(Path(freeze['zip']).read_bytes()).hexdigest()==freeze['zip_sha256']
with zipfile.ZipFile(freeze['zip']) as bundle:
    assert bundle.testzip() is None and len(bundle.namelist())==144
    assert all(bundle.read(name)==(archive/name).read_bytes() for name in bundle.namelist())
source=work.parent/'outputs/t3-public-host-20261005-format-enforced'
bound=json.loads((work.parent/'outputs/neural-cbr-t3-format-host-verification.json').read_text())
assert hashlib.sha256((work.parent/'outputs/neural-cbr-t3-format-host-verification.json').read_bytes()).hexdigest()==protocol['input_verification_sha256']
assert bound['input_artifacts']==protocol['input_artifacts'] and bound['all_recorded_retrieval_traces_verified']
for name,record in protocol['input_artifacts'].items():
    assert Path(record['path']).resolve()==(source/name).resolve()
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==record['sha256']
input_protocol=json.loads((source/'protocol.json').read_text())
assert protocol['host']==input_protocol['host'] and protocol['format_backend']==input_protocol['format_backend']
assert protocol['decode']==input_protocol['decode']
originals={(r['dataset'],r['question_id'],r['condition']):r for r in [json.loads(line) for line in (source/'trials.jsonl').read_text().splitlines()] if r['condition'] in protocol['conditions']}
golds={};files={r['name']:r for r in input_protocol['data']['files']}
for dataset,name,loader in [('squad','squad_dev_v11.json',load_squad),('hotpot','hotpot_dev_distractor_converted.json',load_hotpot)]:
    _,golds[dataset]=loader(work/'t3-public-data-20261005'/name,expected_sha256=files[name]['sha256'],**(dict(allow_unaligned_support=True) if dataset=='hotpot' else {}))
rows=[json.loads(line) for line in (out/'set_audits.jsonl').read_text().splitlines()]
calls=[json.loads(line) for line in (out/'host_calls.jsonl').read_text().splitlines()]
journal=[json.loads(line) for line in (out/'outcome_events.jsonl').read_text().splitlines()]
summary=json.loads((out/'summary.json').read_text())
assert len(rows)==len(originals)==protocol['planned_sets']==summary['sets']==32
assert {(r['dataset'],r['question_id'],r['condition']) for r in rows}==set(originals)
replayed_calls=[];replayed_journal=[];checks=[];credit_count=0;invalid=0;full_repeated_equal=0;full_repeated_different=0
for row in rows:
    key=(row['dataset'],row['question_id'],row['condition']);old=originals[key]
    events=old.get('result',{}).get('events',old.get('partial_events',[]))
    evidence=[c for e in events if e.get('delivered_to_host') for c in e['evidence']]
    assert row['source_event_ids']==[e['event_id'] for e in events] and row['source_error']==old.get('error')
    if not evidence:
        assert row['status']=='unobserved' and not row.get('host_calls');checks.append(dict(key=key,unobserved=True));continue
    assert row['source_model_states']==[e['audit']['model_state_sha256'] for e in events]
    assert row['source_model_versions']==[e['audit']['model_version'] for e in events]
    actual_calls=row['host_calls'];replayed_calls.extend(actual_calls);index=0;captured=[]
    full_payload=json.dumps(evidence,ensure_ascii=False)
    base_prompt=old['host_calls'][-1]['prompt']
    assert old['host_calls'][-1]['stage']=='answer' and base_prompt.count(full_payload)==1
    def evaluate(payload):
        global index,invalid,full_repeated_equal,full_repeated_different
        call=actual_calls[index];index+=1
        assert (call['dataset'],call['question_id'],call['condition'])==key
        assert call['variant_index']==index and call['stage']=='answer' and call['output_tokens']<=128
        assert call['case_ids']==[c['case_id'] for c in payload]
        assert call['prompt']==base_prompt.replace(full_payload,json.dumps(payload,ensure_ascii=False),1)
        global_call_index=len(replayed_calls)-len(actual_calls)+index-1
        assert calls[global_call_index]==call
        if index==1:
            if call['output']==old['host_calls'][-1]['output']:full_repeated_equal+=1
            else:full_repeated_different+=1
        try:
            decision=validate_decision(json.loads(call['output'].strip()),'answer')
            score=hotpot_metrics(decision['answer'],decision['supporting_facts'],golds[key[0]][key[1]]) if key[0]=='hotpot' else answer_metrics(decision['answer'],golds[key[0]][key[1]])
            loss={'answer':1-score['f1']}
            if key[0]=='hotpot':loss.update(support=1-score['sp_f1'],joint=1-score['joint_f1'])
            return dict(losses=loss,decision=decision,scores=score,host_call_index=global_call_index)
        except (ValueError,KeyError,TypeError) as exc:
            invalid+=1
            score=dict(em=0.,f1=0.);loss=dict(answer=1.)
            if key[0]=='hotpot':score.update(sp_f1=0.,joint_f1=0.);loss.update(support=1.,joint=1.)
            return dict(losses=loss,scores=score,task_failure=True,format_error=str(exc),host_call_index=global_call_index)
    fresh=direct_set_audit(evidence,evaluate,outcome_sink=captured.append)
    assert fresh==row['audit'] and index==len(actual_calls)
    replayed_journal.extend(dict(dataset=key[0],question_id=key[1],condition=key[2],observation=r) for r in captured)
    # Independently compute removal signs/credit/complementarity without the
    # maintained utility or credit helper. No missing outcome becomes zero.
    losses={o['variant']:o['outcome']['losses'] for o in row['audit']['outcomes']}
    full=losses['full'];empty=losses['empty' if len(evidence)>1 else f"remove:{evidence[0]['case_id']}"]
    ids=[c['case_id'] for c in evidence]
    activation={r['case_id']:r['weight'] for e in events for r in e['audit']['candidates'] if r['case_id'] in e['audit']['selected_ids']}
    for marginal,credit in zip(fresh['marginal'],row['conditional_credit']):
        cid=marginal['case_id'];removed=losses[f'remove:{cid}']
        assert credit['case_id']==cid and credit['activation_exposure']==activation[cid] and not credit['automatic_update']
        if full is None or removed is None:
            assert marginal['status']==credit['utility_status']=='unobserved' and marginal['loss_differences'] is None and credit['objective_credit'] is None
        else:
            assert marginal['loss_differences']=={k:removed[k]-full[k] for k in full}
            u=removed['answer']-full['answer'];c=activation[cid]*max(u,0);h=activation[cid]*max(-u,0)
            assert credit['objective_credit']==dict(utility=u,C=c,H=h,Q=(c+1)/(c+h+2));credit_count+=1
    expected_gain={k:empty[k]-full[k] for k in full} if full is not None and empty is not None else None
    assert fresh['set_loss_gain']==expected_gain
    expected_pair=None
    if len(ids)==2 and expected_gain is not None and all(losses[f'remove:{cid}'] is not None for cid in ids):
        expected_pair={k:losses[f'remove:{ids[0]}'][k]+losses[f'remove:{ids[1]}'][k]-full[k]-empty[k] for k in full}
    assert fresh['pair_complementarity']==expected_pair
    checks.append(dict(key=key,variants=len(actual_calls),all_checks_passed=True))
assert replayed_calls==calls and replayed_journal==journal
assert len(calls)==summary['host_calls']<=protocol['maximum_calls']
assert sum(c['input_tokens'] for c in calls)==summary['input_tokens']
assert sum(c['output_tokens'] for c in calls)==summary['output_tokens']<=protocol['maximum_generated_tokens']
assert invalid==summary['invalid_outcomes']
report=dict(source=freeze,sets=len(rows),host_calls=len(calls),observed_credit_pairs=credit_count,invalid_outcomes=invalid,
    repeated_full_outputs_equal=full_repeated_equal,repeated_full_outputs_different=full_repeated_different,
    checks=checks,all_recorded_checks_passed=True,
    input_artifacts={name:dict(path=str((out/name).resolve()),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in ('protocol.json','host_calls.jsonl','outcome_events.jsonl','set_audits.jsonl','summary.json')},
    scope='actual recorded frozen-host output/objective/no-refill/prompt/credit/complementarity/journal replay; no fresh LLM generations or global promotion claim')
(work.parent/'outputs/neural-cbr-t3-public-utility-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in {'source','checks','input_artifacts'}}))
