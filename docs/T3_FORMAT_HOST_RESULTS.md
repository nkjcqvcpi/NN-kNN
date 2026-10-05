# Complete constrained public host results

80 trials;144 calls;48,041 input/6,488 output tokens;1430.32s summed trial time;6.36s separate format startup.

| Dataset/condition | EM | Answer F1 | Errors/8 | Calls | Seconds | Input/output tokens | Capped calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| squad/none | 0.2500 | 0.2944 | 0 | 8 | 44.64 | 923/195 | 0 |
| squad/bm25 | 0.1250 | 0.1969 | 2 | 16 | 200.36 | 6531/907 | 1 |
| squad/fixed | 0.1250 | 0.1538 | 3 | 16 | 201.90 | 6228/919 | 2 |
| squad/learned | 0.1250 | 0.1829 | 1 | 16 | 192.67 | 6024/875 | 1 |
| squad/iterative | 0.1250 | 0.2905 | 0 | 16 | 142.56 | 3964/659 | 0 |
| hotpot/none | 0.1250 | 0.1806 | 0 | 8 | 39.79 | 1001/188 | 0 |
| hotpot/bm25 | 0.1250 | 0.2361 | 1 | 16 | 164.43 | 6183/748 | 0 |
| hotpot/fixed | 0.1250 | 0.2232 | 0 | 16 | 149.69 | 6657/671 | 0 |
| hotpot/learned | 0.1250 | 0.2500 | 0 | 16 | 149.02 | 6366/662 | 0 |
| hotpot/iterative | 0.1250 | 0.1607 | 0 | 16 | 145.27 | 4164/664 | 0 |

All 64 actual retrieval events,including seven failed-trial traces,pass full journal/core/payload verification;1,776 manual NN candidates checked. No second retrieval occurs.
Explicit model needs exactly copy the public question in 0/64 requests;the controller does not replace a differing need.
The declared ASCII/lowercase lexical token sequence equals the public question in 20/64 requests;literal punctuation differences do not change this retrieval geometry.
All Hotpot support/joint F1 means remain zero. Error scores stay zero in the denominator.

descriptive engineering comparison; strict validation and decoder both differ from legacy plain; no decoder causal attribution before strict plain control; no QA or multi-hop success claim

Source141 files:90aac35f106efb737679739d58f295e52fd66d44738c72b2e4e686d7b88f7d0e;ZIP530f1eef49900ede6ceaa9b445429cb874330da99ace57b48e031683fddda77f.
Evidence: outputs/neural-cbr-t3-format-host-verification.json;outputs/neural-cbr-t3-format-host-results.json;outputs/t3-public-host-20261005-format-enforced/.

Seven failures comprise four128-token JSON truncations and three invalid ordered
citation pairs at77/98 tokens. The weaker decoder item union does not enforce
title/index order;runtime validation correctly rejects them. Relative to legacy
plain11 errors,the lower seven-error count does not establish a decoder effect:
the strict validation treatment also differs,and the strict plain control is
still running. On SQuAD,fixed NN F1 .1538 is below legacy .4341;Hotpot learned F1
.2500 exceeds legacy .1875,but support/joint F1 remain zero. These tiny
engineering groups do not establish scientific benefits or competitiveness.

Complete immediate journaling is independently established:all64 events replay,
including failures,with no lost error traces. Summed trial time1,430.32s plus6.36s
format preparation are separate from shared model/data loading. Compared with
other runs,machine load and treatment effects confound hardware efficiency.
Actual call/token counts accompany quality;ceilings do not imply equal compute.

New explicit direct-question counterfactuals diagnose the need-input shift:
44/64 host needs change the declared hash vector,14 alter selected ordering and
12 alter set membership.48 fresh actual NN retrievals and16 BM25 query controls
are documented in `T3_NEED_SHIFT_DIAGNOSIS.md`. They leave the actual primary
host-issued needs and experiment outcomes unchanged;no extra LLM answers or
causal answer benefit follows. Training currently uses public question text,
while evaluation receives actual model-generated public needs. This is one
measured input-shift mechanism,not an explanation of all weak QA outcomes.
