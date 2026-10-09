import hashlib
import json
from pathlib import Path

import pytest
import torch

import model.nnknn_rl_workflow as w


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def config(**overrides):
    fields=dict(seed=8,actor_type='nnknn',critic_type='nnknn',total_timesteps=256,
        case_capacity=32,critic_case_capacity=32,top_k=4,policy_update_episodes=1,
        eval_episodes=1,eval_frequency=0,eval_episode_frequency=100000,
        critic_holdout_episode_frequency=0,case_optimizer_maintenance='preserve_by_id',
        case_maintenance_frequency=0,min_cases_per_action=2,case_retention_frequency=64,
        case_retention_queries=8)
    fields.update(overrides)
    return w.make_nnknn_rl_config('smoke',**fields)


def test_live_config_rejects_reset_and_invalid_bounds():
    with pytest.raises(ValueError,match='preserve_by_id'):
        config(case_retention_keep_fraction=.75,case_optimizer_maintenance='reset')
    for fields in ({'case_retention_keep_fraction':float('nan')},{'case_retention_queries':0},
                   {'case_retention_query_budget':-.1},{'case_retention_roles':'MC'}):
        with pytest.raises(ValueError):config(**fields)


def test_direct_dataclass_cannot_bypass_live_constraints(tmp_path):
    output=tmp_path/'invalid'
    with pytest.raises(ValueError,match='preserve_by_id'):
        w.train_nnknn_rl('cartpole',w.NNKNNRLConfig(case_retention_keep_fraction=.9),output_dir=output)
    assert not output.exists()


def test_no_removal_live_observer_preserves_all_learning_states(tmp_path):
    plain=w.train_nnknn_rl('cartpole',config(),output_dir=tmp_path/'plain',device='cpu',progress=False)
    tracked=w.train_nnknn_rl('cartpole',config(case_retention_keep_fraction=1.),output_dir=tmp_path/'tracked',device='cpu',progress=False)
    for role in ('model','value_model','target_model'):assert equal(plain[role].state_dict(),tracked[role].state_dict())
    for name in plain['optimizers']:
        assert equal(w._optimizer_state_snapshot(plain['optimizers'][name]),w._optimizer_state_snapshot(tracked['optimizers'][name]))
    assert plain['final_eval']==tracked['final_eval']
    events=json.loads((tracked['run_dir']/'case_retention_events.json').read_text())['events']
    assert events and any(not e['skipped'] for e in events)
    assert all(e['skipped'] or not e['result']['accepted'] for e in events)


def test_actual_live_compaction_uses_training_only_and_preserves_target_ids(tmp_path):
    state=w.train_nnknn_rl('cartpole',config(case_retention_keep_fraction=.75,
        case_retention_loss_budget=.1,case_retention_query_budget=.1),
        output_dir=tmp_path/'retained',device='cpu',progress=False)
    events=json.loads((state['run_dir']/'case_retention_events.json').read_text())['events']
    assert sum(e['result']['removed'] for e in events if not e['skipped'])>0
    for event in events:
        assert event['reference']['stream'] in {'training_gae','actor_policy_surrogate'}
        assert len(event['reference']['queries'])<=8
        if event['role']=='actor':assert all(event['reference']['policy_ready'])
        if event['skipped']:continue
        assert hashlib.sha256(Path(event['snapshot']).read_bytes()).hexdigest()==event['snapshot_sha256']
        saved=torch.load(event['snapshot'],weights_only=True,map_location='cpu')
        assert saved['optimizer_states']['actor_or_joint']
        result=event['result']
        assert all(end<=start+.1+1e-6 for start,end in zip(result['initial_query_losses'],result['final_query_losses']))
        assert event['alignment']['new_case_ids']==[]
        assert event['alignment']['preserved_case_ids']==result['final_ids']
    assert w._case_store_ids(state['value_model']).tolist()==w._case_store_ids(state['target_model']).tolist()
    restored=w.load_nnknn_rl_checkpoint(state['checkpoint_path'],device='cpu',restore_optimizers=True)
    for role in ('model','value_model','target_model'):assert equal(restored[role].state_dict(),state[role].state_dict())


def test_live_actor_protects_behavior_readiness_under_extreme_requested_compression(tmp_path):
    state=w.train_nnknn_rl('cartpole',config(case_retention_keep_fraction=.01,case_retention_roles='actor',
        case_retention_loss_budget=100.,case_retention_query_budget=None,min_case_entries=8),
        output_dir=tmp_path/'coverage',device='cpu',progress=False)
    events=json.loads((state['run_dir']/'case_retention_events.json').read_text())['events']
    actual=[e for e in events if not e['skipped']]
    assert actual and all(e['result']['target_capacity']>=8 and e['result']['actual_capacity']>=8 for e in actual)
