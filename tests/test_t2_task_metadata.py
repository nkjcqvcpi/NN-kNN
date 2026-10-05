import pytest
import gymnasium as gym

from datasets.rl_tasks import get_rl_task_spec
from model.nnknn_rl_workflow import make_nnknn_rl_config
from model.rl_workflow import _build_early_stopping_tracker


def test_acrobot_real_env_and_return_ceiling_are_distinct_from_time_limit():
    spec=get_rl_task_spec('Acrobot-v1')
    env=gym.make(spec.env_id)
    try:
        obs,_=env.reset(seed=8)
        assert obs.shape==(6,)
        assert env.action_space.n==3
        assert env.spec.max_episode_steps==spec.max_episode_steps==500
        _,reward,_,_,_=env.step(0)
        assert reward==-1.
    finally:
        env.close()
    cfg=make_nnknn_rl_config('smoke')
    assert _build_early_stopping_tracker(cfg,spec).target_score==0.
    assert _build_early_stopping_tracker(cfg,get_rl_task_spec('cartpole')).target_score==500.
    override=make_nnknn_rl_config('smoke',early_stopping_target_score=-100.)
    assert _build_early_stopping_tracker(override,spec).target_score==-100.
