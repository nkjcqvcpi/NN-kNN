
import torch

from model.nnknn_rl_workflow import make_nnknn_rl_config, train_nnknn_rl
from model.t2.runs import RunFiles


def test_bounded_training_audit_preserves_actual_optimization_and_separates_targets(unit_train):
    common=dict(seed=8,actor_type='nnknn',critic_type='nnknn',total_timesteps=256,
        case_capacity=32,critic_case_capacity=32,top_k=4,min_case_entries=2,min_cases_per_action=1,
        eval_episodes=1,eval_episode_frequency=100000,critic_holdout_episode_frequency=0,
        policy_update_episodes=1)
    plain=unit_train('plain',train_nnknn_rl,'cartpole',make_nnknn_rl_config('smoke',**common),
                     device='cpu',progress=False)
    observed=unit_train('observed',train_nnknn_rl,'cartpole',make_nnknn_rl_config('smoke',**common,case_audit_queries_per_batch=2),
                        device='cpu',progress=False)
    for role in ('model','value_model','target_model'):
        before,after=plain[role].state_dict(),observed[role].state_dict()
        assert before.keys()==after.keys()
        assert all(torch.equal(before[k],after[k]) for k in before)
    files=RunFiles(observed['run_dir'])
    events=files.json('case_audit_events.json')['events']
    assert events
    assert {e['stream'] for e in events}=={'training_gae','actor_policy_surrogate'}
    for e in events:
        assert e['phase']=='before_gradient_and_insertion'
        snapshot=torch.load(files.resolve(e['snapshot']),weights_only=True,map_location='cpu')
        assert snapshot['config']['case_audit_queries_per_batch']==2
        assert e['query_index']<2
        if e['role']=='actor':
            assert e['executed_under_ready_policy'] is True
            assert e['advantage_source']=='normalized_clipped_training_gae'
        else:
            assert e['bootstrap_value_source']=='target'
    assert plain['summary']['final_eval']==observed['summary']['final_eval']
