import copy

import pytest
import torch

import model.nnknn_rl_workflow as w


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


@pytest.mark.parametrize('actor_type,critic_type',[('nnknn','nnknn'),('mlp','nnknn'),('nnknn','mlp'),('mlp','mlp')])
def test_forced_earlier_checkpoint_restores_same_optimizer_and_target(monkeypatch,tmp_path,actor_type,critic_type):
    live={}
    build=w._build_actor_critic_models
    target_build=w._build_nnknn_target_value_model
    joint_build=w._build_joint_nnknn_rl_optimizer
    optimizer_build=w._build_nnknn_rl_optimizer
    def models(*args,**kwargs):
        actor,critic=build(*args,**kwargs);live.update(actor=actor,critic=critic)
        return actor,critic
    def target(*args,**kwargs):
        result=target_build(*args,**kwargs);live['target']=result;return result
    def joint(*args,**kwargs):
        result=joint_build(*args,**kwargs);live['actor_optimizer']=result;return result
    def optimizer(model,*args,**kwargs):
        result=optimizer_build(model,*args,**kwargs)
        live['actor_optimizer' if model is live['actor'] else 'critic_optimizer']=result
        return result
    monkeypatch.setattr(w,'_build_actor_critic_models',models)
    monkeypatch.setattr(w,'_build_nnknn_target_value_model',target)
    monkeypatch.setattr(w,'_build_joint_nnknn_rl_optimizer',joint)
    monkeypatch.setattr(w,'_build_nnknn_rl_optimizer',optimizer)
    expected={}
    calls=0
    def evaluate(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==1:
            expected.update(actor=w._model_state(live['actor']),critic=w._copy_state_dict_to_cpu(live['critic']),
                target=w._copy_state_dict_to_cpu(live['target']) if 'target' in live else None,
                optimizers=dict(actor_or_joint=w._optimizer_state_snapshot(live['actor_optimizer']),
                    critic=w._optimizer_state_snapshot(live.get('critic_optimizer'))))
        score=200. if calls==1 else 10.
        return dict(episodes=1,seed=10000,mean_return=score,std_return=0.,min_return=score,
            max_return=score,mean_length=score,episode_metrics=[])
    monkeypatch.setattr(w,'evaluate_nnknn_rl',evaluate)
    cfg=w.make_nnknn_rl_config('smoke',seed=8,actor_type=actor_type,critic_type=critic_type,
        total_timesteps=256,case_capacity=16,critic_case_capacity=16,policy_update_episodes=1,
        eval_frequency=64,eval_episode_frequency=0,eval_episodes=1,critic_holdout_episode_frequency=0)
    state=w.train_nnknn_rl('cartpole',cfg,output_dir=tmp_path/'run',device='cpu',progress=False)
    assert state['summary']['selected_source']=='best_eval'
    assert state['summary']['selected_step']==64
    checkpoint=torch.load(state['checkpoint_path'],weights_only=True,map_location='cpu')
    assert equal(checkpoint['actor_state'],expected['actor'])
    assert equal(checkpoint['critic_state_dict'],expected['critic'])
    assert equal(checkpoint['target_state_dict'],expected['target'])
    assert equal(checkpoint['selected_optimizer_states'],expected['optimizers'])
    restored=w.load_nnknn_rl_checkpoint(state['checkpoint_path'],device='cpu',restore_optimizers=True)
    assert equal(w._model_state(restored['model']),expected['actor'])
    assert equal(restored['value_model'].state_dict(),expected['critic'])
    if expected['target'] is not None:
        assert equal(restored['target_model'].state_dict(),expected['target'])
        assert restored['target_model'].nnknn_model.cases.data_ptr()==restored['value_model'].nnknn_model.cases.data_ptr()
    else:assert restored['target_model'] is None
    for name,opt in restored['optimizers'].items():
        assert equal(w._optimizer_state_snapshot(opt),expected['optimizers'][name])
    # Both original restored and freshly loaded optimizers take the same real step.
    for models_,optimizers in [(state,state['optimizers']),(restored,restored['optimizers'])]:
        for opt in optimizers.values():
            if opt is None:continue
            for group in opt.param_groups:
                for p in group['params']:p.grad=torch.ones_like(p)
            opt.step()
    assert equal(state['model'].state_dict(),restored['model'].state_dict())
    assert equal(state['value_model'].state_dict(),restored['value_model'].state_dict())
    legacy=copy.deepcopy(checkpoint)
    legacy.pop('selected_optimizer_states')
    legacy.pop('target_state_dict')
    legacy_path=tmp_path/'legacy.pt'
    torch.save(legacy,legacy_path)
    historical=w.load_nnknn_rl_checkpoint(legacy_path,device='cpu')
    assert equal(historical['value_model'].state_dict(),expected['critic'])
    assert historical['target_model'] is None and historical['optimizers'] is None
    with pytest.raises(ValueError,match='legacy checkpoint'):
        w.load_nnknn_rl_checkpoint(legacy_path,device='cpu',restore_optimizers=True)
