import json
from pathlib import Path
import sys
import torch

work=Path(__file__).resolve().parent
sys.path.insert(0,str(work/'t2_live_retention_ready_guard_source_archive'))
import model.nnknn_rl_workflow as w
from model.t2.audit import policy_output
torch.set_num_threads(1)
observer=dict(source=json.loads((work/'t2_live_retention_ready_guard_freeze.json').read_text()),runs=[],all_checks_passed=True)
for budget in (2048,8192):
    observer['runs'].extend(json.loads((work/f'acrobot_startup_{budget}'/'manifest.json').read_text())['roles'])
observer['counts']=dict(events=sum(len(json.loads((Path(r['run_dir'])/'case_audit_events.json').read_text())['events']) for r in observer['runs']))
assert len(observer['runs'])==18
def close(a,b):
    assert torch.allclose(torch.as_tensor(a,dtype=torch.float64),torch.as_tensor(b,dtype=torch.float64),atol=2e-5,rtol=2e-5),(a,b)
cache={}
counts=dict(runs=0,snapshots=0,events=0,interventions=0,actor_policy_surrogate=0,training_gae=0,independent_mc=0)
for record in observer['runs']:
    events=json.loads((Path(record['run_dir'])/'case_audit_events.json').read_text())['events']
    with torch.no_grad():
        for event in events:
            path=event['snapshot']
            if path not in cache:
                saved=torch.load(path,weights_only=True,map_location='cpu')
                cfg=w.NNKNNRLConfig(**saved['config'])
                actor,critic=w._build_actor_critic_models(saved['obs_dim'],saved['action_dim'],cfg,torch.device('cpu'))
                w._load_model_state(actor,saved['actor_state']);critic.load_state_dict(saved['critic_state'])
                actor.eval();critic.eval();cache[path]=(actor,critic)
                counts['snapshots']+=1
            actor,critic=cache[path]
            iscritic=event['role']=='critic'
            core=(critic if iscritic else actor).nnknn_model
            q=torch.tensor(event['query']).reshape(1,-1)
            raw=core(q,return_retrieval=True);r=raw[-1]
            indices=r['case_indices'];weights=r['weights'][0]
            assert core.case_ids[indices].tolist()==event['case_ids']
            for actual,stored in [(weights,event['weights']),(r['distances'][0],event['distances']),
                (r['feature_distance_contributions'][0],event['feature_distance_contributions']),
                (core.biases[indices],event['biases']),(core.cases[indices],event['cases']),
                (core.labels[indices],event['labels'])]:close(actual,stored)
            def predict(output):return output.flatten()[0] if iscritic else policy_output(output)[0]
            def loss(value):
                if iscritic:return (value-event['target']).square()
                behavior=(1-event['behavior_epsilon'])*value+event['behavior_epsilon']/value.numel()
                return -event['advantage']*behavior[event['action']].clamp_min(event['probability_floor']).log()
            full=predict(raw[0]);full_loss=loss(full)
            close(full,event['full_prediction']);close(full_loss,event['full_loss'])
            selected=indices[weights>0]
            assert set(i['case_id'] for i in event['interventions'])==set(core.case_ids[selected].tolist())
            for intervention in event['interventions']:
                column=event['case_ids'].index(intervention['case_id'])
                ww=weights.clone();ww[column]=0
                labels=core.labels[indices]
                manual=(ww*labels.flatten()).sum()/ww.sum() if iscritic and ww.sum()>0 else torch.tensor(0.)
                if not iscritic:manual=policy_output((ww@labels).reshape(1,-1))[0]
                mask=torch.zeros(core.case_count(),dtype=torch.bool);mask[selected]=True;mask[indices[column]]=False
                actual=predict(core(q,case_mask=mask)[0]);close(actual,manual)
                removed_loss=loss(actual);delta=float(removed_loss-full_loss)
                close(actual,intervention['removed_prediction']);close(removed_loss,intervention['removed_loss'])
                close(delta,intervention['delta'])
                c,h=max(delta,0.),max(-delta,0.)
                close(c,intervention['C']);close(h,intervention['H'])
                close((c+event['smoothing'])/(c+h+2*event['smoothing']),intervention['Q'])
                counts['interventions']+=1
            counts['events']+=1;counts[event['stream']]+=1
    counts['runs']+=1
    print(f'quality direct {counts["runs"]}/18 {record["role"]}',flush=True)
assert counts['events']==observer['counts']['events'] and all(counts[k]>0 for k in ('actor_policy_surrogate','training_gae'))
(work.parent/'outputs/neural-cbr-t2-startup-training-verification.json').write_text(json.dumps(dict(
    source=observer['source'],counts=counts,all_checks_passed=True,
    scope='all recorded two-training-stream snapshot queries and actual fixed-set removal cores, plus independent renormalized predictions and direct loss credit'),indent=2),encoding='utf-8')
print(json.dumps(counts))
