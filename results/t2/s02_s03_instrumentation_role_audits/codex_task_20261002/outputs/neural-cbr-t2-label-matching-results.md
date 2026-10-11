# T2 critic label matching diagnosis

24 original runs plus24 actual full retrainings. Acrobot,8,192 interactions,three seeds,NN/NN,hard target every batch,128 cases/top-k8,ID-preserved Adam,no scheduled bias pruning,no early stopping. Existing optional label modes are exposed by the pilot CLI; the learner is unchanged.

Actual retraining checks: `{"actual_retrainings": 24, "batches": 225, "training_samples": 196608, "raw_positive_samples": 843, "mutable_calls": 150, "mutable_rows": 5669, "mutable_samples": 36248, "critic_append_samples": 160360, "moment_tensors": 1634, "other_optimizer_states": 4474}`.
Training audits: `{"runs": 24, "snapshots": 201, "events": 468, "interventions": 3744, "actor_policy_surrogate": 66, "training_gae": 402, "independent_mc": 0}`; MC audits: `{"runs": 24, "checkpoints": 48, "events": 192, "interventions": 1536}`.

| Mode | Threshold | Seed | First actual ready behavior | Final action counts | Greedy return | Own-policy MC MSE | Common-policy transfer MSE |
|---|---:|---:|---:|---|---:|---:|---:|
| fixed | 0 | 8 | 7001 | [2, 3, 4] | -500.000 | 4047.966 | 4047.966 |
| mutable | 0 | 8 | 7001 | [2, 3, 4] | -500.000 | 4047.966 | 4047.966 |
| trainable | 0 | 8 | 7001 | [2, 3, 4] | -500.000 | 4062.792 | 4062.709 |
| hybrid | 0 | 8 | 7001 | [2, 3, 4] | -500.000 | 4062.792 | 4062.709 |
| fixed | 0 | 9 | 7001 | [42, 24, 30] | -500.000 | 1014.758 | 1014.758 |
| mutable | 0 | 9 | 7001 | [42, 24, 30] | -500.000 | 1014.758 | 1014.758 |
| trainable | 0 | 9 | 7001 | [45, 16, 61] | -110.667 | 722.686 | 1011.234 |
| hybrid | 0 | 9 | 7001 | [45, 16, 61] | -110.667 | 722.686 | 1011.234 |
| fixed | 0 | 10 | None | [1, 4, 4] | -458.000 | 3981.670 | 3981.670 |
| mutable | 0 | 10 | None | [1, 4, 4] | -458.000 | 3981.670 | 3981.670 |
| trainable | 0 | 10 | None | [1, 4, 4] | -458.000 | 3981.105 | 3981.027 |
| hybrid | 0 | 10 | None | [1, 4, 4] | -458.000 | 3981.105 | 3981.027 |
| mutable | -0.25 | 8 | 8001 | [3, 3, 5] | -500.000 | 3399.651 | 3855.523 |
| hybrid | -0.25 | 8 | 8001 | [3, 3, 5] | -500.000 | 3399.070 | 3854.803 |
| mutable | -0.25 | 9 | 7001 | [54, 23, 45] | -500.000 | 864.228 | 919.668 |
| hybrid | -0.25 | 9 | 7001 | [29, 20, 47] | -370.000 | 975.818 | 920.193 |
| mutable | -0.25 | 10 | None | [2, 2, 6] | -500.000 | 3594.356 | 3496.813 |
| hybrid | -0.25 | 10 | None | [2, 2, 6] | -500.000 | 3593.781 | 3496.196 |
| mutable | -1.0 | 8 | None | [1, 2, 0] | -458.000 | 2338.756 | 2084.347 |
| hybrid | -1.0 | 8 | None | [1, 2, 0] | -458.000 | 2338.394 | 2083.978 |
| mutable | -1.0 | 9 | None | [1, 5, 6] | -458.000 | 1669.109 | 496.260 |
| hybrid | -1.0 | 9 | None | [1, 6, 6] | -458.000 | 1658.435 | 497.896 |
| mutable | -1.0 | 10 | 8001 | [2, 3, 3] | -500.000 | 2075.972 | 1928.450 |
| hybrid | -1.0 | 10 | 8001 | [2, 3, 3] | -500.000 | 2075.651 | 1928.111 |

At threshold0,mutable updates never execute: all three fixed/mutable and three trainable/hybrid checkpoint-state pairs match exactly,including optimizer states. Three fixed runs also reproduce the earlier hard-target startup controls. Trainable labels improve only seed9 to−110.667; this is not a general improvement or default recommendation.

At−.25,mutable/hybrid updates execute. Seed10 becomes ready only after final insertion and has no actual ready behavior during training; its−500 evaluation is prototype policy behavior,not evidence of trained ready-policy gradients. At−1,seeds8/9 never become ready; their−458 evaluation is uniform fallback. Lower critic errors and earlier/greater actor coverage are distinct objectives.

Common-policy rollouts use the baseline actor and baseline time-limit bootstrap,with all eight query/target pairs and target means/sample counts/returns matching exactly. This evaluates each learned V under a different policy and is an off-policy transfer diagnostic,not proof of on-policy calibration. Own-policy MC populations can differ. All diagnostic data are excluded from gradients,case admission and label matching.

Both132-file source archives and ZIP contents/hashes validate; only the pilot CLI differs between the two sources. Existing187-test learner boundary and six focused current optimizer/identity/training-audit tests pass. No Stage A gate,500-case matched-cost comparison,or full-cycle benefit is established.

Source `cb51041421800574e0193514fdc78027528cdb256af10aa11c07f30cb48de347`; ZIP SHA256 `d727a1cf024d1e9c1c2e381ce97a53c65e1bb1831f2338a32d3a5291f67cb922`.
Source `88fafd106c63d461d4e196e9f8e0272a83c2dd3cf7196ac674ef4371c124584f`; ZIP SHA256 `66e3eaecf2342729bd9d5dd002f4536efb77ec05353b0ed36f78438459f53aab`.
