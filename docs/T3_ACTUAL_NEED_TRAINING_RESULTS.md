# Actual-need matching results under common inputs

Six real20-epoch trainings complete:question-trained and actual-need-trained
hard-negative metrics,seeds8/9/10,128 fit/64 article-disjoint tune questions.
Each makes2,560 nonzero updates. Selected epochs are3/4/1 versus3/4/2. The
independent six full retrainings reproduce all selected/final weights,Adam
states,events and curves exactly. Manual float32 gradient error is at most
2.2352e-8;15,360 updates,25,344 evaluation calls and40,704 actual query-identity
checks pass. Question controls also exactly match the old frozen hard-negative
training states. Host and case-specific parameters remain frozen.

Every selected metric is evaluated on both common original questions and common
actual host needs,with the same candidate bank for each question. This avoids
comparing models on different input distributions.2,688 real NN evaluations,
384 BM25 rankings and172,032 independent candidate geometry checks pass.

Tune source recall@2 (provided passage identity,not downstream utility):

| Model | Original question | Actual host need |
| --- | ---: | ---: |
| Fixed lexical geometry | .546875 | .531250 |
| Question-trained seed8 | .656250 | .593750 |
| Actual-need-trained seed8 | .656250 | .625000 |
| Question-trained seed9 | .671875 | .625000 |
| Actual-need-trained seed9 | .656250 | .625000 |
| Question-trained seed10 | .656250 | .625000 |
| Actual-need-trained seed10 | .656250 | .640625 |
| BM25 | 1.000000 | .984375 |

On the same actual needs,actual-need training changes4 misses to hits and2 hits
to misses for seed8;3/3 for seed9;3/2 for seed10. The net changes are2/0/1 of64.
Source recall@1 does not improve:seed8 falls .4375→.421875;seed9/10 remain
.40625/.453125. On original questions,the paired gains/losses are2/2,1/2,3/3.
Thus matching the observed input distribution brings limited recall@2 gains,
not uniform retrieval improvement,and the lexical NN remains far below BM25.
This does not establish competitive semantic retrieval or outcome usefulness.

The two repaired requests retain repeated text. Four fresh BM25 same-bank
direct-question counterfactual rankings keep the supplied source at rank1 for
both requests under both inputs. Fixed NN retains the Canadian-football source,
but the Northwestern question moves from a source hit to a miss under the
repeated actual need. This isolates an observed input robustness failure in
the declared lexical geometry;it does not label the source case harmful or
replace the primary host need. BM25's aggregate one actual-need miss occurs
elsewhere. Format completion should not be equated with semantic need quality.

The64 tune questions are selection data. These findings are exploratory design
diagnostics,not independent confirmation or downstream answer gains. No metric
is promoted globally. Two same-host seed8 public32-trial arms are now running
with identical bounded request grammar/request256/answer128 treatment;their
full audits remain required. The separate internal-interface64-answer experiment
is queued after that job's GPU use.

Evidence:
`outputs/neural-cbr-t3-actual-need-bounded-retraining-verification.json`,
`outputs/neural-cbr-t3-common-need-bounded-evaluation.json`,
`outputs/neural-cbr-t3-actual-need-training-results.json`,
`outputs/neural-cbr-t3-repaired-need-rank-diagnosis.json`,
`work/t3_actual_need_bounded_retraining.log`,`work/t3_bounded_request_256_driver.log`.
All source/data/request bindings use the preserved bounded149-file archive.
