# Matched public outcomes for actual-need training

Both seed8 public32-trial arms and full independent audits complete on the same
existing8 SQuAD/8 Hotpot exploratory questions. They use the same frozen149 source,
9-file host,prompt,decoding,bounded request256/answer128 schema and candidate banks.
The only intended retrieval change is question-trained versus actual-need-trained
hard-negative/recall-selected metric. All32 initial request prompts and raw outputs
are exactly identical. Each arm makes64 actual host calls;all64 combined retrieval
traces and2,368 candidate geometry checks verify,including failure partial events.

| Dataset/condition | Question-trained answer F1 | Actual-need-trained answer F1 | Improved/harmed/tied |
| --- | ---: | ---: | --- |
| SQuAD,two-case one-shot | .159402 | .153846 | 0/1/7 |
| SQuAD,iterative allowance | .415476 | .540476 | 1/0/7 |
| Hotpot,two-case one-shot | .223214 | .223214 | 0/0/8 |
| Hotpot,iterative allowance | .202381 | .202381 | 0/0/8 |

The iterative-allowance arms never actually request a second retrieval:each
receives one case and makes two calls. The SQuAD gain is one question,not evidence
of iterative composition. Two completed SQuAD formatting failures remain in each
arm's one-shot denominator. All Hotpot supporting/joint F1 values remain zero.
Thus the limited tune-source gains do not yield a uniform answer benefit or
demonstrate grounded multi-hop retrieval. Seven selected sets and11 selected
orders change;only six final raw outputs differ across32 paired trials.

Question-trained:21,132 input/2,929 output tokens,551.49 summed trial seconds,
6.23 format initialization seconds. Actual-need-trained:21,090/2,881 tokens,
550.15 summed trial seconds,6.04 initialization seconds. The actual opportunities
match here (64 calls each),but prompt lengths/costs differ with selected evidence;
timing is not a hardware-isolated efficiency result. Training,request collection
and failed prior attempts remain separate costs,not omitted from development.

The original tune selection and these reused16 exploratory questions do not
constitute independent confirmation. Seed8 was fixed before these new outputs,
and neither metric is promoted globally. Source identity is need matching;
case relevance is not automatically utility,reliability or professional truth.
All64 reserved public questions remain unused. The broader competitive role,
typed/scope calibration,feedback/poisoning/rollback,strong host/cloud,human and
qualified validation requirements remain open.

Driver90847 is terminal0. Internal-interface driver96676 has started its actual
four-arm64-answer mechanism experiment on its distinct frozen152 source. New
sublinear/binary CPU fits are also complete;their independent replay and common
input comparison remain pending. These experiments are diagnostic candidates.

Evidence:
`outputs/neural-cbr-t3-public-need-question-control-bounded-verification.json`,
`outputs/neural-cbr-t3-public-need-actual-bounded-verification.json`,
`outputs/neural-cbr-t3-actual-need-public-results.json`,
`outputs/t3-public-host-need-{question-control,actual}-bounded-20261006/`,
`work/t3_bounded_request_256_driver.log`.
