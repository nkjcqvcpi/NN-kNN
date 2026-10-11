import json
from pathlib import Path
import sys
import torch

work=Path(__file__).resolve().parent
archive=work/'t2_selected_checkpoint_source_archive'
sys.path.insert(0,str(archive))
import model.nnknn_rl_workflow as w
torch.set_num_threads(1)
root=Path(r'C:\Users\Administrator\NN-KNN\nnknn-work\results\t2_pi20260920\continuation_20261005\selected_checkpoint')
manifest=json.loads((root/'manifest.json').read_text())
assert len(manifest['roles'])==12
def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
counts=dict(runs=0,nonfinal_phase_selected=0,earlier_environment_step_selected=0,same_step_before_final_update_selected=0,target_states=0,optimizer_states=0,continuation_steps=0)
records=[]
for record in manifest['roles']:
    run=Path(record['run_dir'])
    cp=torch.load(run/'checkpoint.pt',weights_only=True,map_location='cpu')
    live=torch.load(run/'role_audit_state.pt',weights_only=True,map_location='cpu')
    restored=w.load_nnknn_rl_checkpoint(run/'checkpoint.pt',device='cpu',restore_optimizers=True)
    assert equal(w._model_state(restored['model']),cp['actor_state'])
    assert equal(restored['value_model'].state_dict(),cp['critic_state_dict'])
    assert equal(live['critic_state'],cp['critic_state_dict'])
    if cp['target_state_dict'] is not None:
        assert equal(live['target_state'],cp['target_state_dict'])
        assert equal(restored['target_model'].state_dict(),cp['target_state_dict'])
        assert restored['target_model'].nnknn_model.cases.data_ptr()==restored['value_model'].nnknn_model.cases.data_ptr()
        assert restored['target_model'].nnknn_model.labels.data_ptr()!=restored['value_model'].nnknn_model.labels.data_ptr()
        counts['target_states']+=1
    else:assert restored['target_model'] is None and live['target_state'] is None
    twin=w.load_nnknn_rl_checkpoint(run/'checkpoint.pt',device='cpu',restore_optimizers=True)
    for state in (restored,twin):
        for name,optimizer in state['optimizers'].items():
            assert equal(w._optimizer_state_snapshot(optimizer),cp['selected_optimizer_states'][name])
            if optimizer is None:continue
            for group in optimizer.param_groups:
                for p in group['params']:p.grad=torch.ones_like(p)
            optimizer.step()
    assert equal(restored['model'].state_dict(),twin['model'].state_dict())
    assert equal(restored['value_model'].state_dict(),twin['value_model'].state_dict())
    for name in restored['optimizers']:
        assert equal(w._optimizer_state_snapshot(restored['optimizers'][name]),w._optimizer_state_snapshot(twin['optimizers'][name]))
    counts['optimizer_states']+=sum(s is not None for s in cp['selected_optimizer_states'].values())
    counts['continuation_steps']+=1
    if cp['selected_source']=='best_eval':
        assert cp['selected_step']<=cp['actual_timesteps']
        counts['nonfinal_phase_selected']+=1
        counts['earlier_environment_step_selected']+=int(cp['selected_step']<cp['actual_timesteps'])
        counts['same_step_before_final_update_selected']+=int(cp['selected_step']==cp['actual_timesteps'])
    records.append(dict(role=record['role'],selected_source=cp['selected_source'],selected_step=cp['selected_step'],
        actual_timesteps=cp['actual_timesteps'],final_return=record['summary']['last_eval']['mean_return'],
        selected_return=record['summary']['final_eval']['mean_return'],passed=True))
    counts['runs']+=1
    print(f'selected restore {counts["runs"]}/12 {record["role"]}',flush=True)
assert counts['earlier_environment_step_selected']>0
(work.parent/'outputs/neural-cbr-t2-selected-state-verification.json').write_text(json.dumps(dict(
    counts=counts,runs=records,all_checks_passed=True,
    scope='safe selected actor/critic/target/Adam restoration and identical real continuation steps; earlier forced-stage tests cover exact historical state selection; rollout/environment/RNG resumption excluded'),indent=2),encoding='utf-8')
print(json.dumps(counts))
