import torch

from model.nnknn_rl_workflow import compute_gae


def test_negative_reward_zero_value_warmup_has_no_positive_admission():
    # Time-limit boundary ends GAE trace but still permits value bootstrap.
    n=500
    rewards=-torch.ones(n)
    zero=torch.zeros(n)
    boundaries=torch.zeros(n,dtype=torch.bool);boundaries[-1]=True
    terminated=torch.zeros(n,dtype=torch.bool)
    adv,targets=compute_gae(rewards,zero,zero,terminated,gamma=.99,gae_lambda=.95,
                            episode_boundaries=boundaries)
    assert bool((adv<0).all())
    assert torch.equal(adv,targets)
    assert int((adv>0).sum())==0
    # Correct continuing-task constant baseline makes deltas zero, still not
    # evidence for admitting arbitrary action recommendations as helpful.
    calibrated=torch.full((n,),-100.)
    calibrated_adv,_=compute_gae(rewards,calibrated,calibrated,terminated,
        gamma=.99,gae_lambda=.95,episode_boundaries=boundaries)
    assert torch.equal(calibrated_adv,zero)
