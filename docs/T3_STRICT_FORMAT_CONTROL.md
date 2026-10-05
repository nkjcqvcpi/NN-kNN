# Strict plain versus constrained decoder controls

Both80-trial arms use the same strict contract,host,prompts,pools,metric,questions and ceilings. Each makes144 actual host calls. Errors count as zero in every mean.

| Dataset/condition | Plain F1 | Format F1 | Difference | Errors plain/format | Calls | Seconds plain/format | Paired wins/losses/ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| squad/none | 0.2812 | 0.2944 | +0.0132 | 1/0 | 8 | 36.81/44.64 | 1/0/7 |
| squad/bm25 | 0.2019 | 0.1969 | -0.0050 | 4/2 | 16 | 143.14/200.36 | 2/1/5 |
| squad/fixed | 0.2019 | 0.1538 | -0.0481 | 3/3 | 16 | 132.25/201.90 | 1/1/6 |
| squad/learned | 0.1250 | 0.1829 | +0.0579 | 4/1 | 16 | 140.82/192.67 | 2/0/6 |
| squad/iterative | 0.1250 | 0.2905 | +0.1655 | 3/0 | 16 | 131.15/142.56 | 2/0/6 |
| hotpot/none | 0.0556 | 0.1806 | +0.1250 | 1/0 | 8 | 32.24/39.79 | 1/0/7 |
| hotpot/bm25 | 0.2361 | 0.2361 | +0.0000 | 2/1 | 16 | 133.25/164.43 | 0/0/8 |
| hotpot/fixed | 0.0938 | 0.2232 | +0.1295 | 1/0 | 16 | 113.88/149.69 | 2/0/6 |
| hotpot/learned | 0.0625 | 0.2500 | +0.1875 | 1/0 | 16 | 112.60/149.02 | 2/0/6 |
| hotpot/iterative | 0.0357 | 0.1607 | +0.1250 | 4/0 | 16 | 185.93/145.27 | 1/0/7 |

Plain24/80 failures versus constrained7/80. All128 combined retrieval events and3,552 manual NN candidates pass independent journal/core/payload replay;no error trace gaps or second retrievals.
Plain48,219 input/6,543 output tokens,1162.08s summed trials;constrained48,041/6,488 tokens,1430.32s trials plus6.36s format startup. Shared model/data loading is separate.

All144 strict plain raw prompts/outputs/token counts exactly match the legacy plain run. Its13 additional failures result from validation of the same generated content,not changed model generations. Legacy11 missing error traces remain unrecovered.
All Hotpot support/joint F1 means remain zero in both arms.

paired eight engineering questions per dataset/condition; same strict contract/host/prompt/data/ceilings; decoder changes actual needs and answers; costs observed under differing machine load; no confirmation or competitive quality claim

Evidence:outputs/neural-cbr-t3-strict-plain-verification.json;outputs/neural-cbr-t3-format-host-verification.json;outputs/neural-cbr-t3-strict-format-comparison.json.
Shared141-file source90aac35f106efb737679739d58f295e52fd66d44738c72b2e4e686d7b88f7d0e;ZIP530f1eef49900ede6ceaa9b445429cb874330da99ace57b48e031683fddda77f.

The strict comparison isolates enabling the decoder under one validated public
contract. Its total workflow effect includes changing emitted needs,selected
cases and answers;it is not an answer-stage-only intervention. Invalid outputs
are observed task failures,not missing feedback or false-case labels.
Constrained generation reduces failures by17/80 on this engineering sample but
does not guarantee completion:four128-token truncations and three ordered-pair
failures remain. Fixed NN and BM25 SQuAD F1 do not improve under this treatment,
while learned/iterative arms improve over the strict plain counterparts.
Every Hotpot support/joint score remains zero. This supports format reliability,
not established evidence grounding,multi-hop capability or neural competitiveness.

The plain run's13 additional rejections are12 exact-field failures and one extra
ordered-pair failure. All144 raw outputs/prompts/token counts equal legacy plain;
the earlier answer-score change is due to validation of the same content.
The journal repair remains behavior-neutral at that observed boundary. The11
legacy lost traces are not reconstructed by matching raw outputs.

Summed trial time increases from1,162.08 to1,430.32s,plus6.36s format preparation.
Record this observed overhead together with errors/quality/tokens;different
machine load prevents hardware-isolated attribution. All questions and conditions
remain in the denominator;no filtering by successful formatting or answer quality.
The64 reserved public questions remain unused. Direct no-refill utility is now
running on all32 fixed/learned constrained sets,including source failures. The
queued train-only actual-need collection/fitting and stronger/typed/internal/
feedback controls remain distinct unfinished research requirements.
