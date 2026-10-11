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
