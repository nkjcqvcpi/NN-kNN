import hashlib
import json

import torch

import model.nnknn_rl_workflow as w
from model.t2.quality import QualityLedger


def test_quality_and_mc_tracking_preserves_training_and_restores_selected_state(tmp_path,monkeypatch):
    common=dict(seed=8,actor_type='nnknn',critic_type='nnknn',total_timesteps=256,
        case_capacity=32,critic_case_capacity=32,top_k=4,policy_update_episodes=1,
        eval_episodes=1,eval_frequency=0,eval_episode_frequency=100000,
        critic_holdout_episode_frequency=2,critic_holdout_episodes=1,case_audit_queries_per_batch=2)
    plain=w.train_nnknn_rl('cartpole',w.make_nnknn_rl_config('smoke',**common),
        output_dir=tmp_path/'plain',device='cpu',progress=False)
    tracked=w.train_nnknn_rl('cartpole',w.make_nnknn_rl_config('smoke',**common,case_quality_tracking=True),
        output_dir=tmp_path/'tracked',device='cpu',progress=False)
    for role in ('model','value_model','target_model'):
        a,b=plain[role].state_dict(),tracked[role].state_dict()
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    assert plain['summary']['final_eval']==tracked['summary']['final_eval']
    assert {key[:2] for key in tracked['case_quality'].rows}=={
        ('actor','actor_policy_surrogate'),('critic','training_gae'),('critic','independent_mc')}
    restored=w.load_nnknn_rl_checkpoint(tracked['checkpoint_path'],device='cpu')
    assert restored['case_quality'].state_dict()==tracked['case_quality'].state_dict()
    calls=0
    def forced_evaluation(*args,**kwargs):
        nonlocal calls
        calls+=1
        score=200. if calls==1 else 10.
        return dict(mean_return=score,std_return=0.,min_return=score,max_return=score,
                    mean_length=score,episodes=1,seed=10000,episode_metrics=[])
    monkeypatch.setattr(w,'evaluate_nnknn_rl',forced_evaluation)
    common.update(eval_frequency=64,eval_episode_frequency=0)
    selected=w.train_nnknn_rl('cartpole',w.make_nnknn_rl_config('smoke',**common,case_quality_tracking=True),
        output_dir=tmp_path/'selected',device='cpu',progress=False)
    assert selected['summary']['selected_step']==64
    events=json.loads((selected['run_dir']/'case_audit_events.json').read_text())['events']
    expected=QualityLedger(selected['case_quality'].namespace)
    from pathlib import Path
    for event in events:
        assert hashlib.sha256(Path(event['snapshot']).read_bytes()).hexdigest()==event['snapshot_sha256']
        if event['global_step']>64:continue
        event_id=f"{event['role']}:{event['stream']}:{event['global_step']}:{event['query_index']}:{event.get('diagnostic_seed','training')}"
        expected.apply(event,event_id=event_id,active_ids=event['case_ids'])
    assert expected.events and len(expected.events)<len(events)
    assert expected.state_dict()==selected['case_quality'].state_dict()
    checkpoint=torch.load(selected['checkpoint_path'],weights_only=True,map_location='cpu')
    assert checkpoint['selected_case_quality_state']==expected.state_dict()
