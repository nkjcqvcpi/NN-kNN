# Encoder / EMA ablation

Test error (lower is better): 1 - accuracy for classification, train-standardized RMSE for regression. Mean over seeds. `ema_valsel` chooses the EMA momentum per task and setting by validation error.

## retrieval_only

| task | shared | ema090 | ema099 | ema_valsel |
|---|---|---|---|---|
| abalone | 0.7069 | 0.7020 | **0.6991** | **0.6991** |
| airfoil | **0.2351** | 0.3012 | 0.3037 | 0.3012 |
| balance | 0.0624 | **0.0480** | 0.0528 | 0.0528 |
| tabarena_363614 | **0.0154** | 0.0361 | 0.0321 | 0.0321 |
| tabarena_363625 | **0.3771** | 0.3994 | 0.3841 | 0.3841 |
| wine | 0.0556 | 0.0722 | **0.0444** | 0.0722 |

- vs `shared` (tasks, by mean paired difference): ema090: 2 better / 4 worse / 0 tied; ema099: 3 better / 3 worse / 0 tied; ema_valsel: 2 better / 4 worse / 0 tied

## frozen_reuse

| task | shared | ema090 | ema099 | ema_valsel |
|---|---|---|---|---|
| abalone | 0.7007 | 0.6953 | **0.6943** | 0.6953 |
| airfoil | **0.2371** | 0.2922 | 0.2898 | 0.2922 |
| balance | 0.0608 | 0.0464 | **0.0448** | **0.0448** |
| tabarena_363614 | **0.0140** | 0.0287 | 0.0300 | 0.0300 |
| tabarena_363625 | **0.3728** | 0.3929 | 0.3804 | 0.3804 |
| wine | 0.0556 | 0.0722 | **0.0389** | 0.0722 |

- vs `shared` (tasks, by mean paired difference): ema090: 2 better / 4 worse / 0 tied; ema099: 3 better / 3 worse / 0 tied; ema_valsel: 2 better / 4 worse / 0 tied

## alternating_default

| task | shared | ema090 | ema099 | ema_valsel |
|---|---|---|---|---|
| abalone | 0.6976 | 0.6965 | **0.6949** | **0.6949** |
| airfoil | **0.2370** | 0.3012 | 0.3042 | 0.3012 |
| balance | 0.0656 | **0.0496** | 0.0496 | **0.0496** |
| tabarena_363614 | **0.0154** | 0.0334 | 0.0321 | 0.0321 |
| tabarena_363625 | **0.3782** | 0.3985 | 0.3863 | 0.3863 |
| wine | 0.0611 | 0.0444 | **0.0444** | **0.0444** |

- vs `shared` (tasks, by mean paired difference): ema090: 3 better / 3 worse / 0 tied; ema099: 3 better / 3 worse / 0 tied; ema_valsel: 3 better / 3 worse / 0 tied

## joint_summed_w5

| task | shared | ema090 | ema099 | ema_valsel |
|---|---|---|---|---|
| abalone | **0.6900** | 0.6960 | 0.6939 | 0.6939 |
| airfoil | **0.2319** | 0.2866 | 0.3016 | 0.2866 |
| balance | 0.0432 | **0.0352** | 0.0384 | **0.0352** |
| tabarena_363614 | **0.0147** | 0.0207 | 0.0207 | 0.0207 |
| tabarena_363625 | **0.3733** | 0.3750 | 0.3774 | 0.3774 |
| wine | **0.0222** | 0.0500 | 0.0278 | 0.0278 |

- vs `shared` (tasks, by mean paired difference): ema090: 1 better / 5 worse / 0 tied; ema099: 1 better / 5 worse / 0 tied; ema_valsel: 1 better / 5 worse / 0 tied

## Stability during core training (mean over seeds)

| task | arm | drift (mean) | neighbor churn (mean) | churn (last 10 epochs) | online-EMA gap |
|---|---|---|---|---|---|
| abalone | shared | 0.0728 | 0.3250 | 0.2583 |  |
| abalone | ema090 | 0.0445 | 0.2645 | 0.2035 | 0.0236 |
| abalone | ema099 | 0.0130 | 0.2848 | 0.2147 | 0.0531 |
| airfoil | shared | 0.0609 | 0.1319 | 0.0872 |  |
| airfoil | ema090 | 0.0201 | 0.1894 | 0.1461 | 0.0164 |
| airfoil | ema099 | 0.0051 | 0.2129 | 0.1719 | 0.0369 |
| balance | shared | 0.0207 | 0.0865 | 0.0454 |  |
| balance | ema090 | 0.0132 | 0.0788 | 0.0374 | 0.0163 |
| balance | ema099 | 0.0085 | 0.0864 | 0.0418 | 0.1256 |
| tabarena_363614 | shared | 0.0367 | 0.1332 | 0.0585 |  |
| tabarena_363614 | ema090 | 0.0150 | 0.2225 | 0.1479 | 0.0178 |
| tabarena_363614 | ema099 | 0.0050 | 0.2242 | 0.1797 | 0.0633 |
| tabarena_363625 | shared | 0.0332 | 0.1265 | 0.0778 |  |
| tabarena_363625 | ema090 | 0.0190 | 0.1604 | 0.1041 | 0.0177 |
| tabarena_363625 | ema099 | 0.0049 | 0.1760 | 0.1336 | 0.0482 |
| wine | shared | 0.0084 | 0.0702 | 0.0158 |  |
| wine | ema090 | 0.0078 | 0.0751 | 0.0097 | 0.0485 |
| wine | ema099 | 0.0024 | 0.0574 | 0.0058 | 0.1074 |
