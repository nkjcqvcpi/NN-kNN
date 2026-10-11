import hashlib
import json
from pathlib import Path
import sys
import zipfile
import torch

work=Path(__file__).resolve().parent
archive=work/'t2_role_audit_restore_source_archive'
sys.path.insert(0,str(archive))
from model.nnknn_rl_workflow import NNKNNRLConfig, _build_actor_critic_models, NNKNNValueNetwork
from model.t1.artifacts import source_fingerprint
torch.set_num_threads(1)
root=Path(r'C:\Users\Administrator\NN-KNN\nnknn-work\results\t2_pi20260920\continuation_20261004\role_audit_512_restore')
freeze=json.loads((work/'t2_role_audit_restore_freeze.json').read_text())
assert source_fingerprint(archive)==freeze['fingerprint']
assert hashlib.sha256(Path(freeze['zip']).read_bytes()).hexdigest()==freeze['zip_sha256']
with zipfile.ZipFile(freeze['zip']) as z: assert z.testzip() is None and len(z.namelist())==freeze['entries']
manifest=json.loads((root/'manifest.json').read_text())
assert len(manifest['roles'])==12
def close(x,y):
    assert torch.allclose(torch.as_tensor(x,dtype=torch.float64),torch.as_tensor(y,dtype=torch.float64),atol=2e-5,rtol=2e-5),(x,y)
counts=dict(runs=0,checkpoints=0,events=0,interventions=0)
results=[]
for record in manifest['roles']:
    run=Path(record['run_dir'])
    assert record['source_fingerprint']==freeze['fingerprint']
    checkpoint=torch.load(run/'checkpoint.pt',weights_only=True,map_location='cpu')
    saved=torch.load(run/'role_audit_state.pt',weights_only=True,map_location='cpu')
    assert hashlib.sha256((run/'checkpoint.pt').read_bytes()).hexdigest()==record['checkpoint_sha256']
    cfg=NNKNNRLConfig(**saved['config'])
    actor,critic=_build_actor_critic_models(checkpoint['obs_dim'],checkpoint['action_dim'],cfg,torch.device('cpu'))
    critic.load_state_dict(saved['critic_state'])
    events=json.loads((run/'role_audit_events.json').read_text())
    assert len(events)==record['audit_events']
    assert checkpoint['actual_timesteps']==cfg.total_timesteps==512
    if isinstance(critic,NNKNNValueNetwork):
        core=critic.nnknn_model
        assert critic.case_entries==record['summary']['critic_case_entries']>0
        assert torch.equal(critic.active_case_ids(),core.active_case_ids())
        assert len(events)==8
        with torch.no_grad():
            for event in events:
                assert event['stream']=='independent_mc' and event['role']=='critic'
                q=torch.tensor(event['query']).reshape(1,-1)
                raw=core(q,return_retrieval=True)
                r=raw[-1]
                indices=r['case_indices']
                assert core.case_ids[indices].tolist()==event['case_ids']
                close(r['weights'][0],event['weights'])
                close(r['distances'][0],event['distances'])
                close(r['feature_distance_contributions'][0],event['feature_distance_contributions'])
                close(core.biases[indices],event['biases'])
                close(core.labels[indices],event['labels'])
                close(core.cases[indices],event['cases'])
                manual_distance=torch.sqrt(torch.relu(r['feature_distance_contributions'][0].sum(-1)))
                close(manual_distance,event['distances'])
                w=r['weights'][0]
                pred=(w*core.labels[indices].flatten()).sum()
                close(pred,event['full_prediction'])
                close(raw[0].flatten()[0],pred)
                close(event['unmasked_holdout_prediction'],pred)
                full_loss=(raw[0].flatten()[0]-event['target']).square()
                close(full_loss,event['full_loss'])
                selected=indices[w>0]
                assert len(event['interventions'])==len(selected)
                assert set(i['case_id'] for i in event['interventions'])==set(core.case_ids[selected].tolist())
                for intervention in event['interventions']:
                    column=event['case_ids'].index(intervention['case_id'])
                    ww=w.clone(); ww[column]=0
                    pp=(ww*core.labels[indices].flatten()).sum()/ww.sum() if ww.sum()>0 else torch.tensor(0.)
                    mask=torch.zeros(core.case_count(),dtype=torch.bool); mask[selected]=True; mask[indices[column]]=False
                    masked=core(q,case_mask=mask)[0].flatten()[0]
                    close(pp,masked)
                    close(pp,intervention['removed_prediction'])
                    manual_loss=(pp-event['target']).square()
                    loss=(masked-event['target']).square()
                    delta=float(loss-full_loss)
                    manual_delta=float(manual_loss-full_loss)
                    cancellation_bound=64*torch.finfo(torch.float32).eps*(abs(float(loss))+abs(float(full_loss))+1)
                    assert abs(manual_delta-delta)<=cancellation_bound
                    close(loss,intervention['removed_loss'])
                    close(delta,intervention['delta'])
                    c,h=max(delta,0),max(-delta,0)
                    close(c,intervention['C']);close(h,intervention['H'])
                    close((c+event['smoothing'])/(c+h+2*event['smoothing']),intervention['Q'])
                    counts['interventions']+=1
                counts['events']+=1
    else: assert not events
    counts['runs']+=1;counts['checkpoints']+=2
    results.append(dict(role=record['role'],passed=True,mean_return=record['summary']['final_eval']['mean_return'],mc_mse=record['independent_mc']['critic_holdout_mse']))
    print(f'audited {counts["runs"]}/12 {record["role"]}',flush=True)
output=dict(source=freeze,counts=counts,runs=results,all_checks_passed=True,
    scope='safe checkpoint restore; exact active IDs and unmasked retrieval; independently reconstructed fixed-set renormalized predictions/loss/credit and actual masked-core interventions')
(work.parent/'outputs/neural-cbr-t2-role-audit-verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(counts))
