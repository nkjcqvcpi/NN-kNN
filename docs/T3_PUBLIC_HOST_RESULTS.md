# Public host engineering results

80 trials; 144 host calls; 48,219 input tokens; 6,543 output tokens. Errors count as zero in all means.

| Dataset/condition | EM | Answer F1 | Errors | Calls | Seconds | Input/output tokens | Capped calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| squad/none | 0.2500 | 0.2944 | 0 | 8 | 32.37 | 923/192 | 0 |
| squad/bm25 | 0.1250 | 0.3265 | 2 | 16 | 126.44 | 6836/805 | 1 |
| squad/fixed | 0.2500 | 0.4341 | 1 | 16 | 119.00 | 6181/734 | 1 |
| squad/learned | 0.1250 | 0.2421 | 2 | 16 | 125.70 | 5972/778 | 0 |
| squad/iterative | 0.1250 | 0.2905 | 1 | 16 | 114.71 | 4106/720 | 1 |
| hotpot/none | 0.1250 | 0.1806 | 0 | 8 | 28.28 | 1001/182 | 0 |
| hotpot/bm25 | 0.1250 | 0.2361 | 1 | 16 | 116.95 | 6183/757 | 1 |
| hotpot/fixed | 0.1250 | 0.2188 | 0 | 16 | 102.96 | 6368/660 | 0 |
| hotpot/learned | 0.1250 | 0.1875 | 0 | 16 | 100.73 | 6485/647 | 0 |
| hotpot/iterative | 0.0000 | 0.0357 | 4 | 16 | 167.29 | 4164/1068 | 4 |

Supporting/joint Hotpot F1: {'hotpot/none': (0.0, 0.0), 'hotpot/bm25': (0.0, 0.0), 'hotpot/fixed': (0.0, 0.0), 'hotpot/learned': (0.0, 0.0), 'hotpot/iterative': (0.0, 0.0)}

fresh actual core/hash/BM25/manual-feature/event/score/payload replay of 53 recorded retrieval events; 11 post-retrieval error traces missing and not reconstructed; no fresh LLM generations,causal utility or confirmation claim

- eight prospective engineering questions per dataset; no confirmatory inference
- same resource ceilings do not equal actual compute: none uses one call,all other trials use two
- iterative supplies one case and no host requests a second round; not a demonstrated multi-hop loop
- conditional 64-case SQuAD pools and Hotpot distractor contexts,not unrestricted corpus retrieval
- old errored retrieval event IDs missing; new journal cannot reconstruct them

All Hotpot support/joint F1 means are zero. Learned retrieval and iteration show
no established benefit. Higher fixed NN SQuAD F1 on eight questions is exploratory;
EM only equals no retrieval. BM25's stronger tune ranking does not guarantee
small-host answer quality. Need matching,answer quality and inspectable support
remain separate outcomes. The original loss-selected seed8 metric is retained;
later hard-negative weights are not substituted. The64 reserved questions were
not used. Eight calls hit the128-token ceiling;eleven failures comprise ten JSON
parsing errors and one invalid support pair. Formatting alone cannot guarantee
answer truth,support quality or completion within a fixed token ceiling.

Fresh actual core/hash replays verify40 NN events with1,480 manual candidate
distance/contribution comparisons; actual BM25 verifies13 events. Recorded
gold scores,delivered payload prefixes,model/bank/source hashes and144 call token
totals pass. Eleven errored trials reached the second host call but lost their
original retrieval events. The audit reports `all_retrieval_traces_verified=False`;
no original IDs are reconstructed. The committed journal retains events in new
runs even when later generation fails.

Frozen source `bce249f4402c1ca31829143f04c641b479b9d2f387e1dad38fcfbeb29106c393`,
139 files,ZIP `aa49b1825b56c6021e303c1211aa3383c6f099ee4f155ffb18786f84fca7e233`.
Archive/ZIP bytes validate. Task evidence:
`outputs/t3-public-host-20261005-request-v2/{protocol,summary}.json`, `trials.jsonl`,
`outputs/neural-cbr-t3-public-host-verification.json` and
`outputs/neural-cbr-t3-public-host-results.{json,md}`.

The optional strict-schema/token-filtered80-trial run is underway with journaling.
Strict validation also differs from the original plain run; a strict plain
control is required before attributing changes specifically to decoding.
Next separate direct no-refill set-removal utility from ranking,and prospectively
compare the hard-negative checkpoint. Typed heads,calibration,poisoning/rollback,
internal/combined hosts,strong external controls,real participants and complete
T1/T2 gates remain open.
