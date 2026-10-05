# Train-only actual host need preparation

The measured public request shift changes44/64 hash vectors and12 selected sets.
`tools/t3_need_matching.py --need-queries` now fits/evaluates the actual emitted
public needs while keeping original source targets,candidate banks,article split,
training budget,objective,selection and trainable scope fixed. BM25 uses the same
needs. Default original-question behavior remains available as the explicit
matched control. Source identification is need matching,not outcome utility.

`model/t3/need_queries.py` binds the complete prespecified fit/tune questions,
training data/sample hashes,host revision/file manifests,actual emitted JSON,
question identity,request-stage prompt and128-token accounting. An edited stored
need,orphan/duplicate ID,wrong stage,changed data/prompt,dev annotation use or
unobserved request is rejected. Failed or missing needs are not filtered,filled
or replaced by original questions. The fitter refuses incomplete coverage.
Host metadata is a declared binding;actual file/runtime/prompt verification is
provided by the producer and the independent experiment auditor.

`request_prompt` is shared by request collection and future public evaluation.
It preserves the frozen stage-separated instruction literally;the host remains
responsible for emitting the actual public need. No driver planner is introduced.
`tools/t3_train_need_queries.py` is queued to collect all128 fit/64 tune needs
using the same9-file Qwen3-0.6B,format backend,prompt and128-token ceiling as the
audited constrained public run. Only public training questions reach the host;
no answers,passages or dev annotations are supplied. The192-call budget is
charged separately. Generation failures remain logged and reject subsequent fit.

All52 T3 tests pass. A real seed8/one-epoch default-mode training makes128 nonzero
updates and exactly reproduces the old frozen epoch1 selected/final metric,Adam,
initial state,128 training events and two learning-curve rows. Shared prompt
extraction exactly matches the frozen old expression on16 public questions.
This is bounded backwards-compatibility evidence,not completed actual-need
training or a public quality benefit.

The serial driver waits for direct utility generations and their independent
audit,then collects192 requests and runs six20-epoch trainings:original-question
and actual-need hard-negative/recall-selected conditions with seeds8/9/10. Same
prespecified128/64 articles,pools,256 trainable diagonal weights,Adam/projection
and opportunity counts are retained. Fit/tune needs are shared across seeds.
Independent manual-gradient/state replay and prospective public evaluation
remain required. No metric promotion or learned utility claim is made.

Source147 files:
`f3e4de178a6f4bc47bb549eb34a67545107996709b1c8397fa35d425190c8497`,ZIP
`9876cce6fb1768d2120d20fbcc511d556fe7eed9baec4769367e7947d46ad3ee`.
Task evidence: `work/t3_train_need_queries_freeze.json`,
`outputs/neural-cbr-t3-need-query-compat-verification.json`,
`outputs/t3-need-query-default-compat-20261005/`,
`work/run_t3_train_queries_after_utility.ps1`, `work/t3_train_query_driver.log`.
Future outputs: `outputs/t3-train-need-queries-20261005/`,
`outputs/t3-train-need-{question-control,actual}-20261005/`.
