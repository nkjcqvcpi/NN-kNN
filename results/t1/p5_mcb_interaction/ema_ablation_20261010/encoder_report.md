# Encoder / EMA ablation

Test error (lower is better): 1 - accuracy for classification, train-standardized RMSE for regression. Mean over seeds. `ema_valsel` chooses the EMA momentum per task and setting by validation error.

## retrieval_only

| task | raw | shared | ema090 | ema099 | ema0999 | ema_valsel |
|---|---|---|---|---|---|---|
| abalone | 0.7276 | 0.7066 | 0.7055 | **0.7044** | 0.7074 | 0.7074 |
| airfoil | 0.3570 | **0.3147** | 0.3757 | 0.3662 | 0.3362 | 0.3362 |
| balance | 0.1120 | 0.0784 | 0.0688 | **0.0640** | 0.0640 | 0.0688 |
| breast_cancer | 0.0386 | 0.0544 | 0.0386 | **0.0351** | 0.0474 | 0.0474 |
| digits | **0.0183** | 0.0550 | 0.0578 | 0.0700 | 0.0767 | 0.0578 |
| energy_efficiency | **0.0574** | 0.0730 | 0.0961 | 0.0917 | 0.0918 | 0.0961 |
| iris | 0.0600 | 0.0733 | **0.0533** | 0.0667 | 0.0667 | 0.0667 |
| student_performance | **0.4443** | 0.6414 | 0.5525 | 0.5870 | 0.5885 | 0.5870 |
| synthetic_diag | **0.0538** | 0.1623 | 0.1369 | 0.1554 | 0.1415 | 0.1369 |
| wine | **0.0278** | 0.0611 | 0.0389 | 0.0500 | 0.0389 | 0.0500 |
| yacht | **0.1559** | 0.3354 | 0.3137 | 0.3643 | 0.3906 | 0.3137 |
| zebra | **0.1091** | 0.2364 | 0.3455 | 0.4455 | 0.4182 | 0.3455 |
| zebra_special | **0.3500** | 0.4591 | 0.5455 | 0.4727 | 0.5136 | 0.5455 |

- vs `shared` (tasks, by mean paired difference): raw: 10 better / 3 worse / 0 tied; ema090: 8 better / 5 worse / 0 tied; ema099: 7 better / 6 worse / 0 tied; ema0999: 6 better / 7 worse / 0 tied; ema_valsel: 7 better / 6 worse / 0 tied
- vs `raw` (tasks, by mean paired difference): shared: 3 better / 10 worse / 0 tied; ema090: 3 better / 10 worse / 0 tied; ema099: 3 better / 10 worse / 0 tied; ema0999: 3 better / 10 worse / 0 tied; ema_valsel: 3 better / 10 worse / 0 tied

## frozen_reuse

| task | raw | shared | ema090 | ema099 | ema0999 | ema_valsel |
|---|---|---|---|---|---|---|
| abalone | 0.6988 | 0.7058 | 0.7028 | **0.6980** | 0.7020 | 0.7020 |
| airfoil | 0.3481 | **0.3004** | 0.3474 | 0.3378 | 0.3168 | 0.3168 |
| balance | 0.0880 | 0.0752 | 0.0656 | **0.0608** | 0.0624 | 0.0624 |
| breast_cancer | **0.0333** | 0.0544 | 0.0386 | 0.0351 | 0.0474 | 0.0474 |
| digits | **0.0194** | 0.0567 | 0.0578 | 0.0683 | 0.0778 | 0.0578 |
| energy_efficiency | **0.0561** | 0.0712 | 0.0923 | 0.0861 | 0.0805 | 0.0861 |
| iris | 0.0533 | 0.0733 | **0.0467** | 0.0667 | 0.0600 | 0.0600 |
| student_performance | **0.4417** | 0.6464 | 0.5535 | 0.5846 | 0.5868 | 0.5846 |
| synthetic_diag | **0.0546** | 0.1638 | 0.1392 | 0.1538 | 0.1454 | 0.1392 |
| wine | **0.0222** | 0.0667 | 0.0444 | 0.0500 | 0.0389 | 0.0500 |
| yacht | **0.1085** | 0.2810 | 0.2584 | 0.3268 | 0.3765 | 0.2584 |
| zebra | **0.1091** | 0.2545 | 0.3364 | 0.3909 | 0.3818 | 0.3909 |
| zebra_special | **0.3682** | 0.4409 | 0.4591 | 0.4500 | 0.5000 | 0.4591 |

- vs `shared` (tasks, by mean paired difference): raw: 11 better / 2 worse / 0 tied; ema090: 8 better / 5 worse / 0 tied; ema099: 7 better / 6 worse / 0 tied; ema0999: 7 better / 6 worse / 0 tied; ema_valsel: 8 better / 5 worse / 0 tied
- vs `raw` (tasks, by mean paired difference): shared: 2 better / 11 worse / 0 tied; ema090: 3 better / 10 worse / 0 tied; ema099: 3 better / 10 worse / 0 tied; ema0999: 2 better / 11 worse / 0 tied; ema_valsel: 2 better / 11 worse / 0 tied

## alternating_rr

| task | raw | shared | ema090 | ema099 | ema0999 | ema_valsel |
|---|---|---|---|---|---|---|
| abalone | **0.6934** | 0.6989 | 0.7063 | 0.6989 | 0.7005 | 0.7005 |
| airfoil | 0.3493 | **0.3024** | 0.3600 | 0.3493 | 0.3214 | 0.3214 |
| balance | 0.0832 | 0.0800 | 0.0624 | 0.0656 | **0.0560** | 0.0624 |
| breast_cancer | **0.0333** | 0.0561 | 0.0404 | **0.0333** | 0.0474 | 0.0474 |
| digits | **0.0211** | 0.0567 | 0.0572 | 0.0661 | 0.0767 | 0.0572 |
| energy_efficiency | **0.0577** | 0.0976 | 0.0939 | 0.0971 | 0.1017 | 0.0939 |
| iris | 0.0600 | 0.0800 | **0.0467** | 0.0600 | 0.0733 | 0.0733 |
| student_performance | **0.4306** | 0.5959 | 0.5414 | 0.5849 | 0.5846 | 0.5849 |
| synthetic_diag | **0.0546** | 0.1623 | 0.1369 | 0.1485 | 0.1415 | 0.1369 |
| wine | **0.0222** | 0.0500 | 0.0278 | 0.0333 | 0.0389 | 0.0333 |
| yacht | **0.0769** | 0.3265 | 0.2944 | 0.3469 | 0.3863 | 0.2944 |
| zebra | **0.1091** | 0.2091 | 0.3727 | 0.4727 | 0.4182 | 0.3727 |
| zebra_special | **0.3182** | 0.4545 | 0.4682 | 0.4773 | 0.4727 | 0.4682 |

- vs `shared` (tasks, by mean paired difference): raw: 11 better / 2 worse / 0 tied; ema090: 8 better / 5 worse / 0 tied; ema099: 7 better / 6 worse / 0 tied; ema0999: 6 better / 7 worse / 0 tied; ema_valsel: 8 better / 5 worse / 0 tied
- vs `raw` (tasks, by mean paired difference): shared: 2 better / 11 worse / 0 tied; ema090: 2 better / 11 worse / 0 tied; ema099: 2 better / 10 worse / 1 tied; ema0999: 2 better / 11 worse / 0 tied; ema_valsel: 2 better / 11 worse / 0 tied

## joint_rr

| task | raw | shared | ema090 | ema099 | ema0999 | ema_valsel |
|---|---|---|---|---|---|---|
| abalone | 0.6955 | **0.6935** | 0.7076 | 0.6964 | 0.6999 | 0.6999 |
| airfoil | 0.3409 | **0.3001** | 0.3522 | 0.3512 | 0.3228 | 0.3228 |
| balance | 0.0848 | 0.0752 | **0.0512** | 0.0656 | 0.0544 | 0.0544 |
| breast_cancer | **0.0351** | 0.0526 | 0.0421 | **0.0351** | 0.0421 | 0.0421 |
| digits | **0.0228** | 0.0589 | 0.0572 | 0.0667 | 0.0783 | 0.0572 |
| energy_efficiency | **0.0554** | 0.0756 | 0.0965 | 0.0927 | 0.0966 | 0.0965 |
| iris | 0.0600 | 0.0800 | **0.0467** | 0.0600 | 0.0733 | 0.0600 |
| student_performance | **0.4317** | 0.5800 | 0.5369 | 0.5604 | 0.5617 | 0.5604 |
| synthetic_diag | **0.0715** | 0.1654 | 0.1331 | 0.1446 | 0.1546 | 0.1331 |
| wine | **0.0278** | 0.0389 | 0.0333 | **0.0278** | 0.0333 | 0.0333 |
| yacht | **0.0766** | 0.2797 | 0.2986 | 0.3351 | 0.3862 | 0.2986 |
| zebra | **0.1091** | 0.2636 | 0.4000 | 0.4364 | 0.4545 | 0.4000 |
| zebra_special | **0.3136** | 0.4045 | 0.4727 | 0.4727 | 0.4909 | 0.4727 |

- vs `shared` (tasks, by mean paired difference): raw: 10 better / 3 worse / 0 tied; ema090: 7 better / 6 worse / 0 tied; ema099: 6 better / 7 worse / 0 tied; ema0999: 6 better / 7 worse / 0 tied; ema_valsel: 7 better / 6 worse / 0 tied
- vs `raw` (tasks, by mean paired difference): shared: 3 better / 10 worse / 0 tied; ema090: 2 better / 11 worse / 0 tied; ema099: 1 better / 10 worse / 2 tied; ema0999: 2 better / 11 worse / 0 tied; ema_valsel: 2 better / 11 worse / 0 tied

## Stability during core training (mean over seeds)

| task | arm | drift (mean) | neighbor churn (mean) | churn (last 10 epochs) | online-EMA gap |
|---|---|---|---|---|---|
| abalone | raw | 0.0000 | 0.2253 | 0.1651 |  |
| abalone | shared | 0.1869 | 0.3486 | 0.2967 |  |
| abalone | ema090 | 0.0784 | 0.2942 | 0.2523 | 0.0281 |
| abalone | ema099 | 0.0306 | 0.2912 | 0.2402 | 0.0832 |
| abalone | ema0999 | 0.0085 | 0.2770 | 0.2311 | 0.1827 |
| airfoil | raw | 0.0000 | 0.1144 | 0.0901 |  |
| airfoil | shared | 0.1417 | 0.2939 | 0.2658 |  |
| airfoil | ema090 | 0.0361 | 0.2735 | 0.2495 | 0.0246 |
| airfoil | ema099 | 0.0103 | 0.2898 | 0.2513 | 0.0420 |
| airfoil | ema0999 | 0.0026 | 0.2604 | 0.2185 | 0.0859 |
| balance | raw | 0.0000 | 0.1358 | 0.0996 |  |
| balance | shared | 0.0469 | 0.1638 | 0.1039 |  |
| balance | ema090 | 0.0257 | 0.1451 | 0.1098 | 0.0218 |
| balance | ema099 | 0.0131 | 0.1749 | 0.1252 | 0.1257 |
| balance | ema0999 | 0.0045 | 0.1736 | 0.0900 | 0.4418 |
| breast_cancer | raw | 0.0000 | 0.1854 | 0.0605 |  |
| breast_cancer | shared | 0.0257 | 0.1319 | 0.0418 |  |
| breast_cancer | ema090 | 0.0168 | 0.1042 | 0.0532 | 0.0232 |
| breast_cancer | ema099 | 0.0114 | 0.1153 | 0.0290 | 0.1705 |
| breast_cancer | ema0999 | 0.0038 | 0.1115 | 0.0244 | 0.3186 |
| digits | raw | 0.0000 | 0.1363 | 0.1285 |  |
| digits | shared | 0.0616 | 0.1923 | 0.1583 |  |
| digits | ema090 | 0.0528 | 0.1823 | 0.0984 | 0.0180 |
| digits | ema099 | 0.0217 | 0.2044 | 0.1338 | 0.0716 |
| digits | ema0999 | 0.0087 | 0.2018 | 0.1643 | 0.2011 |
| energy_efficiency | raw | 0.0000 | 0.1354 | 0.1171 |  |
| energy_efficiency | shared | 0.0495 | 0.1730 | 0.1274 |  |
| energy_efficiency | ema090 | 0.0253 | 0.1637 | 0.1107 | 0.0127 |
| energy_efficiency | ema099 | 0.0083 | 0.1705 | 0.1299 | 0.0461 |
| energy_efficiency | ema0999 | 0.0028 | 0.1836 | 0.1516 | 0.1399 |
| iris | raw | 0.0000 | 0.0374 | 0.0810 |  |
| iris | shared | 0.0179 | 0.0530 | 0.0446 |  |
| iris | ema090 | 0.0097 | 0.0599 | 0.0566 | 0.0408 |
| iris | ema099 | 0.0041 | 0.0741 | 0.1035 | 0.1375 |
| iris | ema0999 | 0.0007 | 0.0761 | 0.0693 | 0.2076 |
| student_performance | raw | 0.0000 | 0.1353 | 0.1100 |  |
| student_performance | shared | 0.0882 | 0.3138 | 0.3072 |  |
| student_performance | ema090 | 0.0420 | 0.2906 | 0.2345 | 0.0344 |
| student_performance | ema099 | 0.0134 | 0.2870 | 0.2169 | 0.1058 |
| student_performance | ema0999 | 0.0048 | 0.2907 | 0.2299 | 0.3048 |
| synthetic_diag | raw | 0.0000 | 0.1389 | 0.1005 |  |
| synthetic_diag | shared | 0.0731 | 0.2131 | 0.1823 |  |
| synthetic_diag | ema090 | 0.0274 | 0.2020 | 0.1702 | 0.0241 |
| synthetic_diag | ema099 | 0.0087 | 0.2220 | 0.1934 | 0.0674 |
| synthetic_diag | ema0999 | 0.0023 | 0.2284 | 0.2095 | 0.1376 |
| wine | raw | 0.0000 | 0.0408 | 0.0170 |  |
| wine | shared | 0.0107 | 0.0460 | 0.0353 |  |
| wine | ema090 | 0.0075 | 0.0391 | 0.0076 | 0.0263 |
| wine | ema099 | 0.0044 | 0.0429 | 0.0223 | 0.0918 |
| wine | ema0999 | 0.0013 | 0.0502 | 0.0227 | 0.2847 |
| yacht | raw | 0.0000 | 0.0513 | 0.0219 |  |
| yacht | shared | 0.0386 | 0.1000 | 0.0819 |  |
| yacht | ema090 | 0.0194 | 0.0960 | 0.0673 | 0.0226 |
| yacht | ema099 | 0.0082 | 0.1091 | 0.0885 | 0.1367 |
| yacht | ema0999 | 0.0019 | 0.1288 | 0.1059 | 0.3218 |
| zebra | raw | 0.0000 | 0.0426 | 0.0290 |  |
| zebra | shared | 0.0424 | 0.0858 | 0.0767 |  |
| zebra | ema090 | 0.0074 | 0.0525 | 0.0363 | 0.0202 |
| zebra | ema099 | 0.0020 | 0.0625 | 0.0547 | 0.0630 |
| zebra | ema0999 | 0.0003 | 0.0674 | 0.0637 | 0.0996 |
| zebra_special | raw | 0.0000 | 0.0523 | 0.0348 |  |
| zebra_special | shared | 0.0594 | 0.1060 | 0.1109 |  |
| zebra_special | ema090 | 0.0098 | 0.0973 | 0.0767 | 0.0263 |
| zebra_special | ema099 | 0.0027 | 0.1042 | 0.0825 | 0.0587 |
| zebra_special | ema0999 | 0.0004 | 0.1109 | 0.0921 | 0.0772 |
