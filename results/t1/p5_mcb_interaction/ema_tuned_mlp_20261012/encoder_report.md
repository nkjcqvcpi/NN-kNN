# Encoder / EMA ablation

Test error (lower is better): 1 - accuracy for classification, train-standardized RMSE for regression. Mean over seeds. `ema_valsel` chooses the EMA momentum per task and setting by validation error.

## retrieval_only

| task | shared | ema090 | ema099 | ema_valsel |
|---|---|---|---|---|
| abalone | 0.7007 | 0.7007 | **0.6992** | **0.6992** |
| airfoil | **0.2512** | 0.2965 | 0.3125 | 0.2965 |
| balance | 0.0656 | 0.0544 | **0.0464** | **0.0464** |
| tabarena_363614 | **0.0207** | 0.0247 | 0.0281 | 0.0247 |
| tabarena_363625 | **0.3962** | 0.3994 | 0.3982 | 0.3994 |
| wine | **0.0611** | **0.0611** | 0.0722 | **0.0611** |

- vs `shared` (tasks, by mean paired difference): ema090: 2 better / 3 worse / 1 tied; ema099: 2 better / 4 worse / 0 tied; ema_valsel: 2 better / 3 worse / 1 tied

## frozen_reuse

| task | shared | ema090 | ema099 | ema_valsel |
|---|---|---|---|---|
| abalone | 0.6961 | **0.6948** | 0.6960 | 0.6960 |
| airfoil | **0.2490** | 0.2859 | 0.2924 | 0.2859 |
| balance | 0.0608 | 0.0496 | **0.0384** | **0.0384** |
| tabarena_363614 | **0.0147** | 0.0240 | 0.0227 | 0.0240 |
| tabarena_363625 | **0.3842** | 0.3889 | 0.3895 | 0.3889 |
| wine | 0.0611 | **0.0500** | 0.0611 | 0.0611 |

- vs `shared` (tasks, by mean paired difference): ema090: 3 better / 3 worse / 0 tied; ema099: 2 better / 4 worse / 0 tied; ema_valsel: 2 better / 4 worse / 0 tied

## alternating_default

| task | shared | ema090 | ema099 | ema_valsel |
|---|---|---|---|---|
| abalone | 0.6981 | **0.6926** | 0.6957 | 0.6957 |
| airfoil | **0.2490** | 0.2800 | 0.3120 | 0.2800 |
| balance | 0.0640 | **0.0464** | **0.0464** | **0.0464** |
| tabarena_363614 | **0.0220** | 0.0240 | 0.0227 | 0.0240 |
| tabarena_363625 | **0.3822** | 0.3872 | 0.4001 | 0.3872 |
| wine | 0.0611 | **0.0500** | 0.0611 | 0.0611 |

- vs `shared` (tasks, by mean paired difference): ema090: 3 better / 3 worse / 0 tied; ema099: 2 better / 3 worse / 1 tied; ema_valsel: 2 better / 3 worse / 1 tied

## joint_summed_w5

| task | shared | ema090 | ema099 | ema_valsel |
|---|---|---|---|---|
| abalone | 0.7057 | 0.6961 | **0.6934** | 0.6961 |
| airfoil | **0.2378** | 0.2651 | 0.2929 | 0.2651 |
| balance | **0.0384** | 0.0384 | 0.0400 | 0.0384 |
| tabarena_363614 | **0.0127** | 0.0267 | 0.0220 | 0.0220 |
| tabarena_363625 | **0.3551** | 0.3748 | 0.3650 | 0.3650 |
| wine | **0.0278** | 0.0278 | 0.0556 | 0.0556 |

- vs `shared` (tasks, by mean paired difference): ema090: 1 better / 5 worse / 0 tied; ema099: 1 better / 5 worse / 0 tied; ema_valsel: 1 better / 5 worse / 0 tied

## Stability during core training (mean over seeds)

| task | arm | drift (mean) | neighbor churn (mean) | churn (last 10 epochs) | online-EMA gap |
|---|---|---|---|---|---|
| abalone | shared | 0.0844 | 0.3004 | 0.2706 |  |
| abalone | ema090 | 0.1226 | 0.2741 | 0.2030 | 0.0428 |
| abalone | ema099 | 0.0236 | 0.2872 | 0.2278 | 0.0500 |
| airfoil | shared | 0.0674 | 0.2188 | 0.1823 |  |
| airfoil | ema090 | 0.0328 | 0.2424 | 0.2003 | 0.0222 |
| airfoil | ema099 | 0.0092 | 0.2566 | 0.2161 | 0.0327 |
| balance | shared | 0.0268 | 0.1102 | 0.0574 |  |
| balance | ema090 | 0.0193 | 0.1121 | 0.0605 | 0.0122 |
| balance | ema099 | 0.0082 | 0.1015 | 0.0654 | 0.0405 |
| tabarena_363614 | shared | 0.0553 | 0.1536 | 0.0859 |  |
| tabarena_363614 | ema090 | 0.0233 | 0.1878 | 0.0987 | 0.0194 |
| tabarena_363614 | ema099 | 0.0083 | 0.2145 | 0.1397 | 0.0558 |
| tabarena_363625 | shared | 0.0457 | 0.2183 | 0.1444 |  |
| tabarena_363625 | ema090 | 0.0315 | 0.2055 | 0.1358 | 0.0240 |
| tabarena_363625 | ema099 | 0.0077 | 0.2590 | 0.1839 | 0.0423 |
| wine | shared | 0.0092 | 0.0800 | 0.0248 |  |
| wine | ema090 | 0.0087 | 0.0939 | 0.0074 | 0.0365 |
| wine | ema099 | 0.0041 | 0.0880 | 0.0256 | 0.1162 |
