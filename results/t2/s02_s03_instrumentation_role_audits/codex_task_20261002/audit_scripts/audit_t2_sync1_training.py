import hashlib
import json
from pathlib import Path
import sys

import torch

work=Path(__file__).resolve().parent
archive=work/'t2_target_sync1_source_archive'
sys.path.insert(0,str(archive))
from model.nnknn_rl_workflow import NNKNNRLConfig, _build_actor_critic_models, _load_model_state
from model.t2.audit import policy_output
torch.set_num_threads(1)
root=Path(r'C:\Users\Administrator\NN-KNN\nnknn-work\results\t2_pi20260920\continuation_20261004\target_sync1')
oldroot=root.parent/'target_audit_2048'
manifest=json.loads((root/'manifest.json').read_text())
assert len(manifest['roles'])==9
def close(x,y):
    assert torch.allclose(torch.as_tensor(x,dtype=torch.float64),torch.as_tensor(y,dtype=torch.float64),atol=2e-5,rtol=2e-5),(x,y)
def equal(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
counts=dict(runs=0,snapshots=0,critic_gae_events=0,actor_surrogate_events=0,interventions=0,exact_online_reference_pairs=0)
snapshots={}
records=[]
for record in manifest['roles']:
    run=Path(record['run_dir'])
    events=json.loads((run/'case_audit_events.json').read_text())['events']
    assert events
    with torch.no_grad():
        for event in events:
            snapshot=Path(event['snapshot'])
            if str(snapshot) not in snapshots:
                state=torch.load(snapshot,weights_only=True,map_location='cpu')
                cfg=NNKNNRLConfig(**state['config'])
                actor,critic=_build_actor_critic_models(state['obs_dim'],state['action_dim'],cfg,torch.device('cpu'))
                _load_model_state(actor,state['actor_state'])
                critic.load_state_dict(state['critic_state'])
                snapshots[str(snapshot)]=(actor,critic,hashlib.sha256(snapshot.read_bytes()).hexdigest())
                counts['snapshots']+=1
            actor,critic,_=snapshots[str(snapshot)]
            actor.eval();critic.eval()
            iscritic=event['role']=='critic'
            model=critic if iscritic else actor
            core=model.nnknn_model
            q=torch.tensor(event['query']).reshape(1,-1)
            raw=core(q,return_retrieval=True);r=raw[-1]
            ids=core.case_ids[r['case_indices']]
            assert ids.tolist()==event['case_ids']
            close(r['weights'][0],event['weights'])
            close(r['distances'][0],event['distances'])
            close(r['feature_distance_contributions'][0],event['feature_distance_contributions'])
            close(core.labels[r['case_indices']],event['labels'])
            close(core.cases[r['case_indices']],event['cases'])
            def prediction(output): return output.flatten()[0] if iscritic else policy_output(output)[0]
            def loss(value):
                if iscritic: return (value-event['target']).square()
                mixed=(1-event['behavior_epsilon'])*value+event['behavior_epsilon']/value.numel()
                return -event['advantage']*mixed[event['action']].clamp_min(event['probability_floor']).log()
            p=prediction(raw[0]);full_loss=loss(p)
            close(p,event['full_prediction']);close(full_loss,event['full_loss'])
            weights=r['weights'][0]
            indices=r['case_indices']
            selected=indices[weights>0]
            assert set(i['case_id'] for i in event['interventions'])==set(core.case_ids[selected].tolist())
            for intervention in event['interventions']:
                column=event['case_ids'].index(intervention['case_id'])
                ww=weights.clone();ww[column]=0
                labels=core.labels[indices]
                if iscritic:
                    manual=(ww*labels.flatten()).sum()/ww.sum() if ww.sum()>0 else torch.tensor(0.)
                else:
                    manual=policy_output((ww@labels).reshape(1,-1))[0]
                mask=torch.zeros(core.case_count(),dtype=torch.bool);mask[selected]=True;mask[indices[column]]=False
                removed=prediction(core(q,case_mask=mask)[0])
                close(manual,removed);close(removed,intervention['removed_prediction'])
                removed_loss=loss(removed)
                delta=float(removed_loss-full_loss)
                close(removed_loss,intervention['removed_loss']);close(delta,intervention['delta'])
                c,h=max(delta,0.),max(-delta,0.)
                close(c,intervention['C']);close(h,intervention['H'])
                close((c+event['smoothing'])/(c+h+2*event['smoothing']),intervention['Q'])
                counts['interventions']+=1
            assert event['phase']=='before_gradient_and_insertion' and event['query_index']<2
            if iscritic:
                assert event['stream']=='training_gae'
                counts['critic_gae_events']+=1
            else:
                assert event['stream']=='actor_policy_surrogate' and event['executed_under_ready_policy']
                assert event['advantage_source']=='normalized_clipped_training_gae'
                counts['actor_surrogate_events']+=1
    if record['configured_target_mode']=='none':
        old=oldroot/record['task']/'nnknn_nnknn__none'/f's{record["seed"]}'
        oldcp=torch.load(old/'checkpoint.pt',weights_only=True,map_location='cpu')
        current=torch.load(run/'checkpoint.pt',weights_only=True,map_location='cpu')
        assert equal(oldcp['actor_state'],current['actor_state'])
        assert equal(oldcp['critic_state_dict'],current['critic_state_dict'])
        assert equal(oldcp['selected_eval'],current['selected_eval'])
        counts['exact_online_reference_pairs']+=1
    records.append(dict(role=record['role'],training_events=len(events),passed=True))
    counts['runs']+=1
    print(f'audited {counts["runs"]}/9 training {record["role"]}',flush=True)
output=dict(counts=counts,runs=records,all_checks_passed=True,
    snapshot_sha256={k:v[2] for k,v in snapshots.items()},
    scope='all recorded GAE and epsilon-mixed policy surrogate pre-gradient snapshots/events/direct deletions; six exact old/new EMA optimizer-path outputs')
(work.parent/'outputs/neural-cbr-t2-sync1-training-verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(counts))

